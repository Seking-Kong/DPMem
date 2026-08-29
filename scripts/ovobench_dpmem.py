from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata
import torch
import json
import os
import os.path as osp
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import re
import logging
import time
from collections import defaultdict
import argparse
from moviepy.editor import VideoFileClip
import math
import sys
import subprocess
import cv2
from PIL import Image
import tempfile
import shutil

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
from models.qwen3_vl.modeling_qwen3_vl_drop_prev import Qwen3VLForConditionalGeneration

def _b2mb(x: int) -> float:
    return round(x / 1024 / 1024, 2)

def cuda_mem_snapshot():
    """Return per-GPU memory stats in MB; None if CUDA unavailable."""
    if not torch.cuda.is_available():
        return None
    snap = []
    for dev in range(torch.cuda.device_count()):
        snap.append({
            "device": dev,
            "allocated_mb": _b2mb(torch.cuda.memory_allocated(dev)),
            "max_allocated_mb": _b2mb(torch.cuda.max_memory_allocated(dev)),
        })
    return snap

def cuda_reset_peaks():
    if not torch.cuda.is_available():
        return
    for dev in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(dev)

RUN_NAME = "dpmem_ovobench"
DROP_METHOD = 'feature'
DROP_THRESHOLD = 0.5
DROP_ABSOLUTE = True
DROP_MODE = 'both'
CKPT_PATH = "Qwen/Qwen3-VL-8B-Instruct"

TASK_JSON = os.environ.get("OVOBENCH_TASK_JSON", "data/OVO-Bench/ovo_bench_new.json")
VIDEO_DIR = os.environ.get("OVOBENCH_VIDEO_DIR", "data/OVO-Bench/src_videos")
RESULT_DIR = "outputs/ovobench"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"
CHUNK_SUBDIR = "chunked"
MIN_PIXELS = 448*448
MAX_PIXELS = 448*448
MIN_FRAMES = 4
MAX_FRAMES = 720
FPS = 1

FRAMES_PER_CHUNK = 20

prompt_v2t = """You maintain compact structured memory for streaming video QA.

Previous Memory:
{PREV_SUMMARY}

Update the memory using the new 20 frames.

Rules:
- Output ONLY the updated memory.
- Do NOT use decorative adjectives or adverbs.
- Keep factual attributes only: color, number, position, size, shape, state.
- Keep only visible facts useful for future QA: scene, people, objects, actions, spatial relations, state changes.
- Do not infer intention, mood, quality, profession, or background unless visually necessary.
- Merge previous memory with new frames; remove repeated or low-value details.
- Preserve temporal order and important state changes.
- If there is no important change, describe the repeated action once.

Use this format:
[Scene/Objects] ...
[People/Actions] ...
[State/Spatial] ...
[New Change] ...
"""





prompt_v2t_REC = """You maintain compact structured memory for repetitive action counting in a streaming video.

The video may contain people performing the target action repeatedly.
One complete motion counts as one.

Previous Memory:
{PREV_SUMMARY}

Update the memory using the new frames.

Rules:
- Output ONLY the updated memory.
- Focus on counting complete actions.
- Do NOT use decorative adjectives or adverbs.
- Do NOT describe atmosphere, quality, or background unless needed for counting.
- If an action is incomplete, mark it as ongoing, not counted.
- Track different people separately if possible.
- Merge previous memory with new evidence.
- Preserve total count and recent count change.

Use this format:
[Target Action] ...
[People] ...
[Completed Count] total=..., new=...
[Ongoing] ...
[Evidence] ...
[New Change] ...
"""


backward_tasks = ["EPM", "ASI", "HLD"]
realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
forward_tasks = ["REC", "SSR", "CRR"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)

def build_prompt(task, question, options, _anno_, index, context_text=""):
    """
    Build prompt for different tasks.
    context_text: accumulated text context from previous chunks (for streaming)
    """
    context_prefix = ""
    if context_text:
        context_prefix = (
            "Previous Structured Video Memory:\n"
            f"{context_text}\n\n"
        )

    
    if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
        formatted_options = '; '.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(options)) + ';'
        prompt = f"""
        {context_prefix}You have been provided with a text description of previous video content and some frames from the current video segment.Your task is to choose the best answer from the four options provided. 
        Evidence:
        - Previous memory contains past video information.
        - Current frames contain the latest visual evidence.
        General rules:
        - Use current frames as primary evidence for current-state questions.
        - Use previous memory as primary evidence for past-event questions.
        - If previous memory conflicts with current frames, trust the current frames for current-state questions.
        - Do not include any additional text or explanation in your response.
        - Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
        Question:
        {question}
        Options:
        {formatted_options}
        Answer:
        """

    elif task == "REC":
        activity = _anno_["activity"]
        question = "How many times did they " + activity + "?"
        prompt =  f"""{context_prefix}You have been provided with a text description of previous video content and some frames from the current video segment.
        Now you're watching a video in which people may perform a certain type of action repetively. 
        The person performing this kind of action are referred to as 'they' in the following statement.
        You're task is to count how many times have different people in the video perform this kind of action in total.
        Target action:{activity}
        Answer the following question:{question}
        Evidence:
        - Previous memory may contain previous total count and ongoing actions.
        - Current frames may show newly completed or ongoing actions.
        Rules:
        - Count the total number of completed target actions up to the current frames.
        - One complete motion counts as one.
        - Do not count incomplete or ongoing motions.
        - If multiple people perform the action, count all completed actions.
        - Respond with only a single integer.
        - Do not include any additional text or explanation in your response.
        Answer:
        """
    elif task == "SSR":
        step = _anno_["test_info"][index]["step"]
        prompt = f"""
        {context_prefix}You have been provided with a text description of previous video content and some frames from the current video segment.
        You're watching a tutorial video which contain a sequential of steps. 
        The following is one step from the whole procedures: 
        {step}
        Task:
        Determine whether the person is currently performing this step in the current frames.
        Rules:
        - Answer "No" if the step happened only in previous memory, has not started, is already finished, or is not visible.
        - Use previous memory only to understand the procedure order.
        - Do not include any additional text or explanation in your response.
        - Answer only with "Yes" or "No".
        Answer:
        """

    elif task == "CRR":
        question = _anno_["question"]
        answer = _anno_["answer"]
        prompt = f"""
        {context_prefix}You have been provided with a text description of previous video content and some frames from the current video segment.
        You are deciding whether the video has revealed enough visual clues to answer a question.
        The following question are relevant to the end of the video:
        {question}
        Task:
        Decide whether the current frames, especially the latest frames, provide enough information for answering the question.
        Rules:
        - Answer "No" if the answer still requires future frames, unseen content, or guessing.
        - Previous memory can provide context, but it should not replace missing latest-frame evidence.
        - Do not include any additional text or explanation in your response.
        - Answer only with "Yes" or "No".
        Answer:
        """
    return prompt


def chunk_video(video_path, end_time, start_time=0):
    end_time = math.ceil(end_time)
    video_name = osp.splitext(osp.basename(video_path))[0]
    output_dir = osp.join(VIDEO_DIR, CHUNK_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    output_file = osp.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")
    if osp.exists(output_file):
        logger.debug(f"Chunked video {output_file} already exists")
        return output_file

    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    temp_audiofile = osp.join(tmp_dir, f"temp_audio_{video_name}_{start_time}_{end_time}.mp3")

    video = VideoFileClip(video_path)
    clip = None
    try:
        if end_time > video.duration:
            end_time = video.duration
        clip = video.subclip(start_time, end_time)
        clip.write_videofile(output_file, logger=None, temp_audiofile=temp_audiofile)
        logger.debug(f"Chunked video {output_file} saved")
    finally:
        if clip is not None:
            clip.close()
        video.close()

    return output_file


def extract_frames_from_video(video_path, fps=1):
    """
    Extract frames from video at specified fps.
    Returns (frames, original_fps, sampled_frame_indices):
        - frames: list of PIL Images
        - original_fps: the original video fps (float)
        - sampled_frame_indices: list of original frame indices that were sampled
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30
    frame_interval = max(1, int(video_fps / fps))

    frames = []
    sampled_frame_indices = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            sampled_frame_indices.append(frame_count)
        frame_count += 1
    cap.release()
    return frames, video_fps, sampled_frame_indices


def iter_frame_chunks_from_video(video_path, fps=1, frames_per_chunk=20):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30

    frame_interval = max(1, int(video_fps / fps))
    frame_idx = 0

    chunk_frames = []
    chunk_indices = []
    total_sampled = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                chunk_frames.append(Image.fromarray(frame_rgb))
                chunk_indices.append(frame_idx)
                total_sampled += 1

                if len(chunk_frames) >= frames_per_chunk:
                    yield chunk_frames, chunk_indices, video_fps, total_sampled, False
                    chunk_frames = [chunk_frames[-1]]
                    chunk_indices = [chunk_indices[-1]]
            frame_idx += 1

        if chunk_frames:
            yield chunk_frames, chunk_indices, video_fps, total_sampled, True
        elif total_sampled == 0:
            yield [], [], video_fps, 0, True
    finally:
        cap.release()


def save_frames_to_temp(frames, temp_dir, chunk_id):
    """
    Save PIL frames to temp files and return list of paths.
    """
    chunk_dir = osp.join(temp_dir, f"chunk_{chunk_id}")
    os.makedirs(chunk_dir, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        path = osp.join(chunk_dir, f"frame_{i:06d}.jpg")
        frame.save(path, "JPEG")
        paths.append(path)
    return paths


def process_frame_chunk(
    frames_pil,
    text_content,
    processor,
    model,
    min_pixels,
    max_pixels,
    drop_method,
    drop_threshold,
    drop_absolute,
    protect_first_last_frames,
    dr_save_path,
    drop_mode=None,
    is_first_chunk=True,
    video_fps=None,
    frame_indices=None,
    max_new_tokens=256,
):
    """
    Process a chunk of frames (PIL Images) and generate text.
    Returns the generated response text.
    For Qwen3-VL: pass PIL images and VideoMetadata for accurate timestamps.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames_pil,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {
                    "type": "text",
                    "text": text_content
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    video_meta = VideoMetadata(
        total_num_frames=len(frames_pil),
        fps=video_fps,
        frames_indices=frame_indices if frame_indices is not None else list(range(len(frames_pil))),
    )
    inputs = processor(
        text=[text],
        videos=[frames_pil],
        video_metadata=[video_meta],
        do_sample_frames=False,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        drop_method=drop_method,
        drop_threshold=drop_threshold,
        drop_absolute=drop_absolute,
        protect_first_last_frames=protect_first_last_frames,
        dr_save_path=dr_save_path,
        drop_mode=drop_mode,
        is_first_chunk=is_first_chunk,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def process_video_streaming(
    video_path,
    final_prompt,
    processor,
    model,
    min_pixels,
    max_pixels,
    fps,
    frames_per_chunk,
    drop_method,
    drop_threshold,
    drop_absolute,
    dr_save_path,
    drop_mode=None,
    task=None,
):
    """
    Process video using streaming approach.
    Returns (response, total_frames, num_chunks, cuda_mem)
    """
    temp_dir = tempfile.mkdtemp()

    try:
        context_text = ""
        response = None
        num_chunks = 0
        total_frames = 0
        got_any_frames = False

        cuda_reset_peaks()

        for frame_list, indices_list, video_fps, total_sampled, is_last_chunk in iter_frame_chunks_from_video(
            video_path, fps=fps, frames_per_chunk=frames_per_chunk
        ):
            if len(frame_list) == 0 and total_sampled == 0:
                logger.error(f"No frames extracted from video: {video_path}")
                return None, 0, 0, None

            got_any_frames = True
            total_frames = total_sampled
            num_chunks += 1

            if is_last_chunk:
                message_text = final_prompt.replace("{CONTEXT_TEXT}", context_text if context_text else "")
                response = process_frame_chunk(
                    frames_pil=frame_list,
                    text_content=message_text,
                    processor=processor,
                    model=model,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    drop_method=drop_method,
                    drop_threshold=drop_threshold,
                    drop_absolute=drop_absolute,
                    protect_first_last_frames=True,
                    dr_save_path=dr_save_path,
                    drop_mode=drop_mode,
                    is_first_chunk=(num_chunks == 1),
                    video_fps=video_fps,
                    frame_indices=indices_list,
                    max_new_tokens=MAX_NEW_TOKENS_FINAL,
                )
            else:
                prev_summary = context_text if context_text else ""
                if task == "REC":
                    message_text = prompt_v2t_REC.format(PREV_SUMMARY=prev_summary)
                else:
                    message_text = prompt_v2t.format(PREV_SUMMARY=prev_summary)

                response = process_frame_chunk(
                    frames_pil=frame_list,
                    text_content=message_text,
                    processor=processor,
                    model=model,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    drop_method=drop_method,
                    drop_threshold=drop_threshold,
                    drop_absolute=drop_absolute,
                    protect_first_last_frames=True,
                    dr_save_path=dr_save_path,
                    drop_mode=drop_mode,
                    is_first_chunk=(num_chunks == 1),
                    video_fps=video_fps,
                    frame_indices=indices_list,
                    max_new_tokens=MAX_NEW_TOKENS_INTERMEDIATE,
                )
                context_text = response

            del frame_list, indices_list

        if not got_any_frames:
            logger.error(f"No frames extracted from video: {video_path}")
            return None, 0, 0, None

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cuda_mem = cuda_mem_snapshot()

        return response, total_frames, num_chunks, cuda_mem

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def score(results):
    def calculate_score_backward_realtime(results):
        def get_score(response, gt):
            if response == None:
                return 0
            return int(gt in response)
        for i in range(len(results)):
            results[i]["score"] = get_score(results[i]["response"], results[i]["ground_truth"])
        
        scores = {}
        for i in range(len(results)):
            if not results[i]["task"] in scores.keys():
                scores[results[i]["task"]] = [results[i]["score"]]
            else:
                scores[results[i]["task"]].append(results[i]["score"])
        return results, scores

    def calculate_score_forward(results):
        def get_score_REC(response, gt):
            if response == None:
                return 0
            import re
            response = re.findall(r'\d+', response)
            response = "".join(response)
            return response == str(gt)
        
        def get_score_SSR_CRR(response, gt):
            if response == None:
                return 0
            return int(gt in response)
        
        scores = {}
        tasks = list(set([result["task"] for result in results]))
        for task in tasks:
            scores[task] = []
        for i, result in enumerate(results):
            if result["task"] == "REC":
                cnt_correct = 0
                for j, test_info_ in enumerate(result["test_info"]):
                    cnt_correct += get_score_REC(test_info_["response"], test_info_["count"])
                scores["REC"].append(cnt_correct / len(result["test_info"]))
            if result["task"] == "SSR":
                cnt_correct = 0
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    cnt_correct += get_score_SSR_CRR(test_info_["response"], gt)
                scores["SSR"].append(cnt_correct / len(result["test_info"]))
            if result["task"] == "CRR":
                cnt_correct = 0
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    cnt_correct += get_score_SSR_CRR(test_info_["response"], gt)
                scores["CRR"].append(cnt_correct / len(result["test_info"]))
        return results, scores
    
    backward_results = results["backward"]
    realtime_results = results["realtime"]
    forward_results = results["forward"]
    avg_scores = {
        "backward": [],
        "realtime": [],
        "forward": []
    }

    if len(backward_results) > 0:
        backward_results, backward_scores = calculate_score_backward_realtime(backward_results)
        for k, v in backward_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
            avg_scores["backward"].append(sum(v)/len(v))
        logger.info(f"Backward Avg.: {100 * sum(avg_scores['backward'])/len(avg_scores['backward']):.2f}\n")
    else:
        pass
        
    if len(realtime_results) > 0:
        realtime_results, realtime_scores = calculate_score_backward_realtime(realtime_results)
        for k, v in realtime_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
            avg_scores["realtime"].append(sum(v)/len(v))
        logger.info(f"Realtime Avg.: {100 * sum(avg_scores['realtime'])/len(avg_scores['realtime']):.2f}\n")
    else:
        pass

    if len(forward_results) > 0:
        forward_results, forward_scores = calculate_score_forward(forward_results)
        for k, v in forward_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
            avg_scores["forward"].append(sum(v)/len(v))
        logger.info(f"Forward Avg.: {100 * sum(avg_scores['forward'])/len(avg_scores['forward']):.2f}\n")
    else:
        pass

    logger.info(f"Total Avg.: {100 * (sum(avg_scores['backward'])/len(avg_scores['backward']) + sum(avg_scores['realtime'])/len(avg_scores['realtime']) + sum(avg_scores['forward'])/len(avg_scores['forward'])) / 3:.2f}")
    
    return avg_scores


def parse_gpu_ids(gpu_ids_arg: str):
    if gpu_ids_arg:
        return [x.strip() for x in gpu_ids_arg.split(",") if x.strip()]
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        return [x.strip() for x in cuda_visible.split(",") if x.strip()]
    return ["0"]


def split_tasks_by_chunk(task_list, num_chunks, chunk_idx):
    if num_chunks <= 1:
        return task_list
    return [item for i, item in enumerate(task_list) if i % num_chunks == chunk_idx]


def merge_jsonl_files(src_paths, dst_path):
    with open(dst_path, "w") as fout:
        for src in src_paths:
            if not osp.exists(src):
                continue
            with open(src, "r") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--drop_method", type=str, default=DROP_METHOD)
    parser.add_argument("--drop_threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--drop_relative", action="store_true")
    parser.add_argument("--drop_mode", type=str, default=DROP_MODE,
                        choices=["both", "first_only", "last_only", "neither"],
                        help="Boundary-frame drop policy: both=mode3 (default, "
                             "current logic), first_only=mode1, last_only=mode2, "
                             "neither=mode4")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--task_json", type=str, default=TASK_JSON)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--frames_per_chunk", type=int, default=FRAMES_PER_CHUNK)
    parser.add_argument("--max_new_tokens_intermediate", type=int, default=512, help="max_new_tokens for non-final (streaming summary) chunks")
    parser.add_argument("--max_new_tokens_final", type=int, default=64, help="max_new_tokens for final (non-streaming) chunks")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of parallel workers (= number of GPUs)")
    parser.add_argument("--chunk_idx", type=int, default=-1, help="Current chunk index, -1 means launcher mode")
    parser.add_argument("--gpu_ids", type=str, default="", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--gpus_per_worker", type=int, default=1, help="Number of GPUs per worker")
    parser.add_argument("--run_id", type=str, default="", help="Shared run id across workers")
    parser.add_argument("--worker_mode", action="store_true", help="Internal flag used by launcher")
    args = parser.parse_args()

    if args.num_chunks > 1 and args.chunk_idx < 0 and not args.worker_mode:
        run_id = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
        launcher_start_time = time.time()
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        gpw = args.gpus_per_worker
        needed = args.num_chunks * gpw

        shared_mode = needed > len(gpu_ids)
        if shared_mode:
            if gpw != 1:
                raise ValueError(
                    f"Multi-worker-per-GPU (shared) mode requires --gpus_per_worker=1 "
                    f"(got gpw={gpw}, num_chunks={args.num_chunks}, "
                    f"gpus={len(gpu_ids)}). Reduce num_chunks or set gpus_per_worker=1."
                )
            if len(gpu_ids) == 0:
                raise ValueError("No GPU found in CUDA_VISIBLE_DEVICES")

        os.makedirs(args.result_dir, exist_ok=True)
        os.makedirs(osp.join(args.result_dir, "output"), exist_ok=True)
        os.makedirs(osp.join(args.result_dir, "drop"), exist_ok=True)
        os.makedirs(osp.join(args.result_dir, "log"), exist_ok=True)

        launcher_log_path = osp.join(args.result_dir, f"log/{args.run_name}_{run_id}.log")
        launcher_file_handler = logging.FileHandler(launcher_log_path)
        launcher_file_handler.setFormatter(fmt)
        launcher_file_handler.setLevel(logging.INFO)
        logger.addHandler(launcher_file_handler)
        logger.info(f"Running {args.run_name} on OVO-Bench (launcher mode)")
        if shared_mode:
            logger.info(
                f"num_chunks={args.num_chunks}, gpus_per_worker={gpw}, "
                f"physical_gpu_ids={','.join(gpu_ids)}, "
                f"workers_per_gpu={args.num_chunks / max(1, len(gpu_ids)):.1f} (shared mode)"
            )
        else:
            logger.info(f"num_chunks={args.num_chunks}, gpus_per_worker={gpw}, gpu_ids={','.join(gpu_ids[:needed])}")

        procs = []
        for idx in range(args.num_chunks):
            if shared_mode:
                worker_gpu_ids = [gpu_ids[idx % len(gpu_ids)]]
            else:
                worker_gpu_ids = gpu_ids[idx * gpw : (idx + 1) * gpw]
            worker_env = os.environ.copy()
            worker_env["CUDA_VISIBLE_DEVICES"] = ",".join(worker_gpu_ids)
            cmd = [
                sys.executable, __file__,
                "--run_name", args.run_name,
                "--drop_method", args.drop_method,
                "--drop_threshold", str(args.drop_threshold),
                "--drop_mode", args.drop_mode,
                "--ckpt_path", args.ckpt_path,
                "--task_json", args.task_json,
                "--video_dir", args.video_dir,
                "--result_dir", args.result_dir,
                "--fps", str(args.fps),
                "--min_pixels", str(args.min_pixels),
                "--max_pixels", str(args.max_pixels),
                "--min_frames", str(args.min_frames),
                "--max_frames", str(args.max_frames),
                "--frames_per_chunk", str(args.frames_per_chunk),
                "--num_chunks", str(args.num_chunks),
                "--chunk_idx", str(idx),
                "--gpu_ids", args.gpu_ids,
                "--gpus_per_worker", str(gpw),
                "--run_id", run_id,
                "--worker_mode",
            ]
            if args.drop_relative:
                cmd.append("--drop_relative")
            logger.info(f"Launch chunk {idx}/{args.num_chunks} on GPU {','.join(worker_gpu_ids)}")
            procs.append(subprocess.Popen(cmd, env=worker_env))

        return_codes = [p.wait() for p in procs]
        if any(code != 0 for code in return_codes):
            logger.error(f"Some worker processes failed, return codes: {return_codes}")
            raise RuntimeError(f"Some worker processes failed, return codes: {return_codes}")

        merged_output = osp.join(args.result_dir, f"output/{args.run_name}_{run_id}.jsonl")
        merged_drop = osp.join(args.result_dir, f"drop/{args.run_name}_{run_id}.jsonl")
        output_parts = [
            osp.join(args.result_dir, f"output/{args.run_name}_chunk{idx}_{run_id}.jsonl")
            for idx in range(args.num_chunks)
        ]
        drop_parts = [
            osp.join(args.result_dir, f"drop/{args.run_name}_chunk{idx}_{run_id}.jsonl")
            for idx in range(args.num_chunks)
        ]
        merge_jsonl_files(output_parts, merged_output)
        merge_jsonl_files(drop_parts, merged_drop)
        logger.info(f"Merged outputs -> {merged_output}")
        logger.info(f"Merged drops   -> {merged_drop}")

        results = defaultdict(list)
        with open(merged_output, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item['task'] in backward_tasks:
                    results['backward'].append(item)
                elif item['task'] in realtime_tasks:
                    results['realtime'].append(item)
                elif item['task'] in forward_tasks:
                    results['forward'].append(item)
        avg_scores = score(results)

        if args.drop_method and args.drop_method != "none" and osp.exists(merged_drop):
            drop_list, total_list, ratio_list = [], [], []
            with open(merged_drop, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    drop_ratio_info = json.loads(line)
                    drop_list.append(drop_ratio_info['drop'])
                    total_list.append(drop_ratio_info['total'])
                    ratio_list.append(drop_ratio_info['ratio'])
            if len(total_list) > 0 and sum(total_list) > 0:
                total_dr = sum(drop_list) / sum(total_list)
                avg_dr = sum(ratio_list) / len(ratio_list)
                logger.info(f"Total drop ratio (weighted drop ratio): {100*total_dr:.1f}%")
                logger.info(f"Average drop ratio (unweighted drop ratio): {100*avg_dr:.1f}%")

        launcher_cost_time = int(time.time() - launcher_start_time)
        logger.info(
            f"Inference cost time: {launcher_cost_time // 3600}h "
            f"{(launcher_cost_time % 3600) // 60}m {launcher_cost_time % 60}s"
        )

        logger.info(f"Launcher log -> {launcher_log_path}")
        sys.exit(0)

    curr_time = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.chunk_idx >= 0:
        run_name_with_chunk = f"{args.run_name}_chunk{args.chunk_idx}"
        CHUNK_SUBDIR = f"chunked_{args.num_chunks}_{args.chunk_idx}"
    else:
        run_name_with_chunk = args.run_name

    RUN_NAME = args.run_name
    DROP_METHOD = args.drop_method
    DROP_THRESHOLD = args.drop_threshold
    DROP_ABSOLUTE = not args.drop_relative
    DROP_MODE = args.drop_mode
    CKPT_PATH = args.ckpt_path
    RESULT_DIR = args.result_dir
    TASK_JSON = args.task_json
    VIDEO_DIR = args.video_dir
    LOG_PATH = osp.join(RESULT_DIR, f"log/{run_name_with_chunk}_{curr_time}.log")
    OUTPUT_JSONL = osp.join(RESULT_DIR, f"output/{run_name_with_chunk}_{curr_time}.jsonl")
    DR_SAVE_PATH = osp.join(RESULT_DIR, f"drop/{run_name_with_chunk}_{curr_time}.jsonl")
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    MIN_FRAMES = args.min_frames
    MAX_FRAMES = args.max_frames
    FPS = args.fps
    MAX_NEW_TOKENS_INTERMEDIATE = args.max_new_tokens_intermediate
    MAX_NEW_TOKENS_FINAL = args.max_new_tokens_final
    FRAMES_PER_CHUNK = args.frames_per_chunk

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'output'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'drop'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'log'), exist_ok=True)

    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.info(f"Running {RUN_NAME} on OVO-Bench")
    logger.info(f"Drop method: {DROP_METHOD}")
    logger.info(f"Drop threshold: {DROP_THRESHOLD}")
    logger.info("Drop absolute" if DROP_ABSOLUTE else "Drop relative")
    logger.info(f"Drop mode: {DROP_MODE}")
    logger.info(f"Checkpoint path: {CKPT_PATH}")
    logger.info(f"Result dir: {RESULT_DIR}")
    logger.info(f"Task json: {TASK_JSON}")
    logger.info(f"Video dir: {VIDEO_DIR}")
    logger.info(f"Output jsonl: {OUTPUT_JSONL}")
    logger.info(f"Drop ratio info save path: {DR_SAVE_PATH}")
    logger.info(f"Min pixels: {MIN_PIXELS}")
    logger.info(f"Max pixels: {MAX_PIXELS}")
    logger.info(f"Max frames: {MAX_FRAMES}")
    logger.info(f"Min frames: {MIN_FRAMES}")
    logger.info(f"FPS: {FPS}")
    logger.info(f"Frames per chunk: {FRAMES_PER_CHUNK}")
    if args.chunk_idx >= 0:
        logger.info(f"Worker mode: chunk_idx={args.chunk_idx}, num_chunks={args.num_chunks}")
    logger.info(f"Run id: {curr_time}")

    if osp.exists(OUTPUT_JSONL):
        os.remove(OUTPUT_JSONL)
    if osp.exists(DR_SAVE_PATH):
        os.remove(DR_SAVE_PATH)

    torch.manual_seed(1236)
    logger.info(f"Set manual seed to 1234")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CKPT_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        CKPT_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    logger.info(f"Load model and processor from {CKPT_PATH}")

    with open(TASK_JSON, 'r') as f:
        all_task_list = json.load(f)
    if args.chunk_idx >= 0:
        task_list = split_tasks_by_chunk(all_task_list, args.num_chunks, args.chunk_idx)
        logger.info(f"Chunk split enabled: chunk_idx={args.chunk_idx}, num_chunks={args.num_chunks}, samples={len(task_list)}")
    else:
        task_list = all_task_list
        logger.info(f"Total tasks: {len(task_list)}")

    import traceback as tb_module
    start_time = time.time()
    for item in tqdm(task_list):
        try:
            if item['task'] in backward_tasks or item['task'] in realtime_tasks:
                id, video, task, question, options, realtime, gt = \
                    item['id'], item['video'], item['task'], item['question'], item['options'], item['realtime'], item['gt']

                final_prompt = build_prompt(
                    task=task,
                    question=question,
                    options=options,
                    _anno_=None,
                    index=None,
                    context_text="{CONTEXT_TEXT}",
                )

                video_path = osp.join(VIDEO_DIR, video)
                chunk_video_path = chunk_video(video_path=video_path, end_time=realtime)

                response, total_frames, num_chunks, cuda_mem = process_video_streaming(
                    video_path=chunk_video_path,
                    final_prompt=final_prompt,
                    processor=processor,
                    model=model,
                    min_pixels=MIN_PIXELS,
                    max_pixels=MAX_PIXELS,
                    fps=FPS,
                    frames_per_chunk=FRAMES_PER_CHUNK,
                    drop_method=DROP_METHOD,
                    drop_threshold=DROP_THRESHOLD,
                    drop_absolute=DROP_ABSOLUTE,
                    dr_save_path=DR_SAVE_PATH,
                    drop_mode=DROP_MODE,
                    task=task,
                )

                if response is None:
                    logger.error(f"Failed to process video: {chunk_video_path}")
                    output_dict = {
                        'id': id,
                        'video': video,
                        'task': task,
                        'question': question,
                        'response': None,
                        'ground_truth': chr(65 + gt),
                        'error': 'no_frames_extracted',
                    }
                else:
                    output_dict = {
                        'id': id,
                        'video': video,
                        'task': task,
                        'question': question,
                        'response': response,
                        'ground_truth': chr(65 + gt),
                        'total_frames': total_frames,
                        'num_chunks': num_chunks,
                        'cuda_mem': cuda_mem,
                    }

            elif item['task'] in forward_tasks:
                id, video, task, test_info = \
                    item['id'], item['video'], item['task'], item['test_info']
                for i in range(len(test_info)):
                    try:
                        final_prompt = build_prompt(
                            task=task,
                            question=None,
                            options=None,
                            _anno_=item,
                            index=i,
                            context_text="{CONTEXT_TEXT}",
                        )

                        realtime = test_info[i]['realtime']
                        video_path = osp.join(VIDEO_DIR, video)
                        chunk_video_path = chunk_video(video_path=video_path, end_time=realtime)

                        response, total_frames, num_chunks, cuda_mem = process_video_streaming(
                            video_path=chunk_video_path,
                            final_prompt=final_prompt,
                            processor=processor,
                            model=model,
                            min_pixels=MIN_PIXELS,
                            max_pixels=MAX_PIXELS,
                            fps=FPS,
                            frames_per_chunk=FRAMES_PER_CHUNK,
                            drop_method=DROP_METHOD,
                            drop_threshold=DROP_THRESHOLD,
                            drop_absolute=DROP_ABSOLUTE,
                            dr_save_path=DR_SAVE_PATH,
                            drop_mode=DROP_MODE,
                            task=task,
                        )

                        item['test_info'][i]['response'] = response
                        item['test_info'][i]['total_frames'] = total_frames
                        item['test_info'][i]['num_chunks'] = num_chunks
                        item['test_info'][i]['cuda_mem'] = cuda_mem
                    except Exception as e:
                        logger.error(f"Error in forward sub-item {i} of id={id}: {e}")
                        logger.error(tb_module.format_exc())
                        item['test_info'][i]['response'] = None
                        item['test_info'][i]['error'] = str(e)

                output_dict = item

            with open(OUTPUT_JSONL, 'a') as f:
                f.write(json.dumps(output_dict) + '\n')
        except Exception as e:
            logger.error(f"Error in processing id={item.get('id', '?')}: {e}")
            logger.error(tb_module.format_exc())
            error_dict = {
                'id': item.get('id'),
                'video': item.get('video'),
                'task': item.get('task'),
                'response': None,
                'error': str(e),
            }
            with open(OUTPUT_JSONL, 'a') as f:
                f.write(json.dumps(error_dict) + '\n')
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    end_time = time.time()
    cost_time = int(end_time - start_time)

    if args.chunk_idx >= 0:
        logger.info(f"Chunk {args.chunk_idx}/{args.num_chunks} finished in {cost_time}s")
        sys.exit(0)

    results = defaultdict(list)
    with open(OUTPUT_JSONL, 'r') as f:
        lines = f.readlines()
    for line in lines:
        item = json.loads(line)
        if item['task'] in backward_tasks:
            results['backward'].append(item)
        elif item['task'] in realtime_tasks:
            results['realtime'].append(item)
        elif item['task'] in forward_tasks:
            results['forward'].append(item)
    avg_scores = score(results)

    if DROP_METHOD is not None and DR_SAVE_PATH is not None and osp.exists(DR_SAVE_PATH):
        drop_list, total_list, ratio_list = [], [], []
        with open(DR_SAVE_PATH, 'r') as f:
            lines = f.readlines()
        for line in lines:
            drop_ratio_info = json.loads(line)
            drop_list.append(drop_ratio_info['drop'])
            total_list.append(drop_ratio_info['total'])
            ratio_list.append(drop_ratio_info['ratio'])
        if len(total_list) > 0 and sum(total_list) > 0:
            total_dr = sum(drop_list) / sum(total_list)
            avg_dr = sum(ratio_list) / len(ratio_list)
            logger.info(f"Total drop ratio (weighted drop ratio): {100*total_dr:.1f}%")
            logger.info(f"Average drop ratio (unweighted drop ratio): {100*avg_dr:.1f}%")

    logger.info(f"Inference cost time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")


