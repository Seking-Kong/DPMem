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
import ffmpeg
import sys
import cv2
import subprocess
from PIL import Image
import tempfile
import shutil

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
"""
Changed the pruning target from the current frame to the preceding frame.
"""
from models.qwen3_vl.modeling_qwen3_vl_dpmem import Qwen3VLForConditionalGeneration


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

RUN_NAME = "dpmem_streamingbench"
DROP_METHOD = 'feature'
DROP_THRESHOLD = 0.5
DROP_ABSOLUTE = True
DROP_MODE = 'both'
CKPT_PATH = "Qwen/Qwen3-VL-8B-Instruct"

TASK_CSV = os.environ.get("STREAMINGBENCH_TASK_CSV", "data/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv")
VIDEO_DIR = os.environ.get("STREAMINGBENCH_VIDEO_DIR", "data/StreamingBench/Real-Time Visual Understanding")
RESULT_DIR = "outputs/streamingbench"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"
MIN_PIXELS = 448*448
MAX_PIXELS = 448*448
MIN_FRAMES = 4
MAX_FRAMES = 1016

FRAMES_PER_CHUNK = 20
SAMPLING_FPS = 1








prompt_v2t = """You maintain compact structured memory for streaming video QA.

Previous Memory:
{PREV_SUMMARY}

Update the memory using the new frames.

Rules:
- Output ONLY the updated memory.
- Do not use decorative adjectives or adverbs, such as professional, perfect, clearly, beautiful.
- Keep adjectives only for factual attributes: color, number, position, size, visible state.
- Keep only visible facts useful for future QA.
- Merge previous memory with new frames. Remove low-value details.
- Preserve important temporal order and state changes.
- If there is no important change, briefly state the repeated action once.Remove  irrelevant details.

Use this format:
[Scene/Objects] ...
[People/Actions] ...
[Spatial/State] ...
[New Change] ...
"""


prompt_qa = """You are an advanced video question-answering AI assistant.

Previous Structured Video Memory:
{}

You are given:
1. A compact memory of previous video content.
2. Current video frames.
3. A multiple-choice question.

Combine the previous video description with the current video frames.
Use the current frames as the primary evidence.
Use the previous memory only as temporal context.
If the memory conflicts with the current frames, trust the current frames.

Choose the best answer. Respond with only the letter of the correct option: A, B, C, or D.

Question: {}

Options:
{}

The best option is:"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)

def time_to_seconds(time_str):
    if len(time_str) == 5:
        time_obj = datetime.strptime(time_str, '%M:%S')
    else:
        time_obj = datetime.strptime(time_str, '%H:%M:%S')
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return total_seconds

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]

def split_video(video_file, start_time, end_time):
    """
    Split video into prefix part based on timestamp.
    video_file: path to video file
    start_time: start time in seconds
    end_time: end time in seconds
    """
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(os.path.dirname(video_file), "tmp_60")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")
    if os.path.exists(output_file):
        logger.debug(f"Video file {output_file} already exists.")
        return output_file
    (
        ffmpeg
        .input(video_file, ss=int(start_time))
        .output(output_file, t=(int(end_time) - int(start_time)), vcodec='libx264', acodec='aac')
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    logger.debug(f"Video: {output_file} splitting completed.")
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
    
    For Qwen3-VL, we pass PIL Images directly as a video to the processor.
    We also provide VideoMetadata so Qwen3-VL knows the original video fps
    and the exact frame indices for accurate timestamp calculation.
    
    Args:
        video_fps: original video fps (float). Used to construct VideoMetadata.
        frame_indices: list of original frame indices for the frames in this chunk.
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


def parse_gpu_ids(gpu_ids_arg: str):
    if gpu_ids_arg:
        return [x.strip() for x in gpu_ids_arg.split(",") if x.strip()]
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        return [x.strip() for x in cuda_visible.split(",") if x.strip()]
    return ["0"]


def merge_jsonl_files(src_paths, dst_path):
    with open(dst_path, "w") as fout:
        for src in src_paths:
            if not osp.exists(src):
                continue
            with open(src, "r") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)


def log_summary_from_files(output_jsonl, drop_jsonl, logger_obj):
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    if osp.exists(output_jsonl):
        with open(output_jsonl, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                cnt_total["overall"] += 1
                cnt_total[item["task_type"]] += 1
                if extract_characters_regex(item["response"]) == item["answer"]:
                    cnt_correct["overall"] += 1
                    cnt_correct[item["task_type"]] += 1
    else:
        logger_obj.info(f"Output file not found: {output_jsonl}")

    task_types = [
        "Object Perception",
        "Causal Reasoning",
        "Clips Summarize",
        "Attribute Perception",
        "Event Understanding",
        "Text-Rich Understanding",
        "Prospective Reasoning",
        "Spatial Understanding",
        "Action Perception",
        "Counting",
    ]
    for task_type in task_types:
        if cnt_total[task_type] == 0:
            logger_obj.info(f"- {task_type}: No question processed")
        else:
            logger_obj.info(
                f"- {task_type}: {cnt_correct[task_type]}/{cnt_total[task_type]} = "
                f"{100 * cnt_correct[task_type] / cnt_total[task_type]:.2f}%"
            )
    if cnt_total["overall"] == 0:
        logger_obj.info("No question processed")
    else:
        logger_obj.info(
            f"Total: {cnt_total['overall']}, Correct: {cnt_correct['overall']}, "
            f"Accuracy: {100 * cnt_correct['overall'] / cnt_total['overall']:.2f}%"
        )

    if DROP_METHOD is not None and drop_jsonl is not None and osp.exists(drop_jsonl):
        drop_list, total_list, ratio_list = [], [], []
        with open(drop_jsonl, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                drop_ratio_info = json.loads(line)
                drop_list.append(drop_ratio_info["drop"])
                total_list.append(drop_ratio_info["total"])
                ratio_list.append(drop_ratio_info["ratio"])
        if total_list and sum(total_list) > 0 and ratio_list:
            total_dr = sum(drop_list) / sum(total_list)
            avg_dr = sum(ratio_list) / len(ratio_list)
            logger_obj.info(f"Total drop ratio (weighted drop ratio): {100 * total_dr:.1f}%")
            logger_obj.info(f"Average drop ratio (unweighted drop ratio): {100 * avg_dr:.1f}%")

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
    parser.add_argument("--task_csv", type=str, default=TASK_CSV)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--frames_per_chunk", type=int, default=FRAMES_PER_CHUNK)
    parser.add_argument("--max_new_tokens_intermediate", type=int, default=512,
                        help="max_new_tokens for non-final (streaming summary) chunks")
    parser.add_argument("--max_new_tokens_final", type=int, default=64,
                        help="max_new_tokens for the final (QA) chunk")
    parser.add_argument("--fps",type=float,default=SAMPLING_FPS)
    parser.add_argument("--num_chunks", type=int, default=1, help="Dataset is split into num_chunks pieces")
    parser.add_argument("--chunk_idx", type=int, default=-1, help="Current chunk index, -1 means launcher mode")
    parser.add_argument("--gpu_ids", type=str, default="", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--gpus_per_worker", type=int, default=1, help="Number of GPUs per worker (e.g. 2 means each worker uses 2 GPUs)")
    parser.add_argument("--run_id", type=str, default="", help="Shared run id across workers")
    parser.add_argument("--worker_mode", action="store_true", help="Internal flag used by launcher process")
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
        launcher_log_path = osp.join(args.result_dir, LOG_PATH.format(run_name=args.run_name, curr_time=run_id))
        launcher_file_handler = logging.FileHandler(launcher_log_path)
        launcher_file_handler.setFormatter(fmt)
        launcher_file_handler.setLevel(logging.INFO)
        logger.addHandler(launcher_file_handler)
        logger.info(f"Running {args.run_name} on StreamingBench (launcher mode)")
        if shared_mode:
            logger.info(
                f"num_chunks={args.num_chunks}, gpus_per_worker={gpw}, "
                f"physical_gpu_ids={','.join(gpu_ids)}, "
                f"workers_per_gpu={args.num_chunks / max(1, len(gpu_ids)):.1f} (shared mode)"
            )
        else:
            logger.info(f"num_chunks={args.num_chunks}, gpus_per_worker={gpw}, gpu_ids={','.join(gpu_ids[:needed])}")
        logger.info(f"Merged output target: output/{args.run_name}_{run_id}.jsonl")

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
                "--task_csv", args.task_csv,
                "--video_dir", args.video_dir,
                "--result_dir", args.result_dir,
                "--min_pixels", str(args.min_pixels),
                "--max_pixels", str(args.max_pixels),
                "--min_frames", str(args.min_frames),
                "--max_frames", str(args.max_frames),
                "--frames_per_chunk", str(args.frames_per_chunk),
                "--fps", str(args.fps),
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
            raise RuntimeError(f"Some worker processes failed, return codes: {return_codes}")

        merged_output = osp.join(args.result_dir, f"output/{args.run_name}_{run_id}.jsonl")
        merged_drop = osp.join(args.result_dir, f"drop/{args.run_name}_{run_id}.jsonl")
        output_parts = [osp.join(args.result_dir, f"output/{args.run_name}_chunk{idx}_{run_id}.jsonl") for idx in range(args.num_chunks)]
        drop_parts = [osp.join(args.result_dir, f"drop/{args.run_name}_chunk{idx}_{run_id}.jsonl") for idx in range(args.num_chunks)]
        merge_jsonl_files(output_parts, merged_output)
        merge_jsonl_files(drop_parts, merged_drop)
        print(f"Merged outputs -> {merged_output}")
        print(f"Merged drops   -> {merged_drop}")

        log_summary_from_files(merged_output, merged_drop, logger)
        launcher_cost_time = int(time.time() - launcher_start_time)
        logger.info(
            f"Inference cost time: {launcher_cost_time // 3600}h "
            f"{(launcher_cost_time % 3600) // 60}m {launcher_cost_time % 60}s"
        )
        print(f"Merged log     -> {launcher_log_path}")
        sys.exit(0)

    curr_time = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.chunk_idx >= 0:
        run_name_with_chunk = f"{args.run_name}_chunk{args.chunk_idx}"
    else:
        run_name_with_chunk = args.run_name
    RUN_NAME = args.run_name
    DROP_METHOD = args.drop_method
    DROP_THRESHOLD = args.drop_threshold
    DROP_ABSOLUTE = not args.drop_relative
    DROP_MODE = args.drop_mode
    CKPT_PATH = args.ckpt_path
    RESULT_DIR = args.result_dir
    TASK_CSV = args.task_csv
    VIDEO_DIR = args.video_dir
    LOG_PATH = osp.join(RESULT_DIR, LOG_PATH.format(run_name=run_name_with_chunk, curr_time=curr_time))
    OUTPUT_JSONL = osp.join(RESULT_DIR, OUTPUT_JSONL.format(run_name=run_name_with_chunk, curr_time=curr_time))
    DR_SAVE_PATH = osp.join(RESULT_DIR, DR_SAVE_PATH.format(run_name=run_name_with_chunk, curr_time=curr_time))
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    MIN_FRAMES = args.min_frames
    MAX_FRAMES = args.max_frames
    FRAMES_PER_CHUNK = args.frames_per_chunk
    SAMPLING_FPS = args.fps
    MAX_NEW_TOKENS_INTERMEDIATE = args.max_new_tokens_intermediate
    MAX_NEW_TOKENS_FINAL = args.max_new_tokens_final

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'output'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'drop'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'log'), exist_ok=True)
    
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    logger.info(f"Running {RUN_NAME} on StreamingBench")
    logger.info(f"Drop method: {DROP_METHOD}")
    logger.info(f"Drop threshold: {DROP_THRESHOLD}")
    logger.info("Drop absolute" if DROP_ABSOLUTE else "Drop relative")
    logger.info(f"Drop mode: {DROP_MODE}")
    logger.info(f"Checkpoint path: {CKPT_PATH}")
    logger.info(f"Result dir: {RESULT_DIR}")
    logger.info(f"Task csv: {TASK_CSV}")
    logger.info(f"Video dir: {VIDEO_DIR}")
    logger.info(f"Output jsonl: {OUTPUT_JSONL}")
    logger.info(f"Drop ratio info save path: {DR_SAVE_PATH}")
    logger.info(f"Min pixels: {MIN_PIXELS}")
    logger.info(f"Max pixels: {MAX_PIXELS}")
    logger.info(f"Max frames: {MAX_FRAMES}")
    logger.info(f"Min frames: {MIN_FRAMES}")
    logger.info(f"Frames per chunk: {FRAMES_PER_CHUNK}")
    logger.info(f"FPS: {SAMPLING_FPS}")
    
    torch.manual_seed(1235)
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

    
    task_df = pd.read_csv(TASK_CSV)
    if args.chunk_idx >= 0:
        task_df = task_df.iloc[args.chunk_idx::args.num_chunks].reset_index(drop=True)
        logger.info(f"Chunk split enabled: chunk_idx={args.chunk_idx}, num_chunks={args.num_chunks}, samples={len(task_df)}")
    
    start_time = time.time()
    for row in tqdm(task_df.itertuples(), total=len(task_df)):
        try:
            question_id, task_type, question, time_stamp, answer, options, frames_required, temporal_clue_type = \
                row.question_id, row.task_type, row.question, row.time_stamp, row.answer, row.options, row.frames_required, row.temporal_clue_type
            video_path = osp.join(VIDEO_DIR, f"sample_{question_id.split('_')[-2]}", "video.mp4")
            time_stamp_sec = time_to_seconds(time_stamp)
            video_path = split_video(video_path, 0, time_stamp_sec)
            
            all_frames, video_fps, all_frame_indices = extract_frames_from_video(video_path, SAMPLING_FPS)
            
            if len(all_frames) == 0:
                logger.error(f"No frames extracted from video: {video_path}")
                continue
            
            temp_dir = tempfile.mkdtemp()
            
            try:
                frame_list = []
                indices_list = []
                context_text = ""
                
                cuda_reset_peaks()
                
                frame_idx = 0
                chunk_id = 0
                total_frames = len(all_frames)
                
                while frame_idx < total_frames:
                    frame_list.append(all_frames[frame_idx])
                    indices_list.append(all_frame_indices[frame_idx])
                    frame_idx += 1
                    
                    is_last_chunk = (frame_idx >= total_frames)
                    
                    if len(frame_list) == FRAMES_PER_CHUNK or is_last_chunk:
                        if is_last_chunk:
                            options_text = '\n'.join(eval(options))
                            message_text = prompt_qa.format(context_text, question, options_text)

                            response = process_frame_chunk(
                                frames_pil=frame_list,
                                text_content=message_text,
                                processor=processor,
                                model=model,
                                min_pixels=MIN_PIXELS,
                                max_pixels=MAX_PIXELS,
                                drop_method=DROP_METHOD,
                                drop_threshold=DROP_THRESHOLD,
                                drop_absolute=DROP_ABSOLUTE,
                                protect_first_last_frames=True,
                                dr_save_path=DR_SAVE_PATH,
                                drop_mode=DROP_MODE,
                                is_first_chunk=(chunk_id == 0),
                                video_fps=video_fps,
                                frame_indices=indices_list,
                                max_new_tokens=MAX_NEW_TOKENS_FINAL,
                            )

                        else:
                            prev_summary = context_text if context_text else " "
                            message_text = prompt_v2t.format(PREV_SUMMARY=prev_summary)
                            
                            response = process_frame_chunk(
                                frames_pil=frame_list,
                                text_content=message_text,
                                processor=processor,
                                model=model,
                                min_pixels=MIN_PIXELS,
                                max_pixels=MAX_PIXELS,
                                drop_method=DROP_METHOD,
                                drop_threshold=DROP_THRESHOLD,
                                drop_absolute=DROP_ABSOLUTE,
                                protect_first_last_frames=True,
                                dr_save_path=DR_SAVE_PATH,
                                drop_mode=DROP_MODE,
                                is_first_chunk=(chunk_id == 0),
                                video_fps=video_fps,
                                frame_indices=indices_list,
                                max_new_tokens=MAX_NEW_TOKENS_INTERMEDIATE,
                            )
                            
                            context_text = response


                            frame_list = [frame_list[-1]]
                            indices_list = [indices_list[-1]]
                            chunk_id += 1
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                cuda_mem = cuda_mem_snapshot()

                
                easy_output_dict = {
                    'question_id': question_id,
                    'task_type': task_type,
                    'answer': answer,
                    'response': response,
                    'total_frames': total_frames,
                    'num_chunks': chunk_id + 1,
                    'cuda_mem': cuda_mem,
                }

                with open(OUTPUT_JSONL, 'a' if osp.exists(OUTPUT_JSONL) else 'w') as f:
                    f.write(json.dumps(easy_output_dict) + '\n')
                    
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        except Exception as e:
            logger.error(f"Error in processing {row}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    end_time = time.time()
    cost_time = int(end_time - start_time)
    
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    if osp.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if not line.strip():
                continue
            item = json.loads(line)
            cnt_total['overall'] += 1
            cnt_total[item['task_type']] += 1
            if extract_characters_regex(item['response']) == item['answer']:
                cnt_correct['overall'] += 1
                cnt_correct[item['task_type']] += 1
    else:
        logger.info(f"Output file not found: {OUTPUT_JSONL}")
    task_types = ['Object Perception', 'Causal Reasoning', 'Clips Summarize', 'Attribute Perception', 'Event Understanding', 'Text-Rich Understanding', 'Prospective Reasoning', 'Spatial Understanding', 'Action Perception', 'Counting']
    for task_type in task_types:
        if cnt_total[task_type] == 0:
            logger.info(f"- {task_type}: No question processed")
        else:
            logger.info(f"- {task_type}: {cnt_correct[task_type]}/{cnt_total[task_type]} = {100*cnt_correct[task_type]/cnt_total[task_type]:.2f}%")
    if cnt_total['overall'] == 0:
        logger.info("No question processed")
    else:
        logger.info(f"Total: {cnt_total['overall']}, Correct: {cnt_correct['overall']}, Accuracy: {100*cnt_correct['overall']/cnt_total['overall']:.2f}%")
    
    if DROP_METHOD is not None and DR_SAVE_PATH is not None and osp.exists(DR_SAVE_PATH):
        drop_list, total_list, ratio_list = [], [], []
        with open(DR_SAVE_PATH, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if not line.strip():
                continue
            drop_ratio_info = json.loads(line)
            drop_list.append(drop_ratio_info['drop'])
            total_list.append(drop_ratio_info['total'])
            ratio_list.append(drop_ratio_info['ratio'])
        if total_list and sum(total_list) > 0 and ratio_list:
            total_dr = sum(drop_list) / sum(total_list)
            avg_dr = sum(ratio_list) / len(ratio_list)
            logger.info(f"Total drop ratio (weighted drop ratio): {100*total_dr:.1f}%")
            logger.info(f"Average drop ratio (unweighted drop ratio): {100*avg_dr:.1f}%")
    
    logger.info(f"Inference cost time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")
    
