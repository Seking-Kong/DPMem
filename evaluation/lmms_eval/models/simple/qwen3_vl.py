import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

import cv2
import torch
import torch.distributed
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


# ---------------------------------------------------------------------------
#  Streaming DPMem prompts (question-aware)
# ---------------------------------------------------------------------------
PROMPT_V2T = """You maintain compact structured memory for streaming video QA.

You are given a question and its options. Your goal is to preserve information from the video frames that is relevant to answering this question.

Question and Options:
{QUESTION_CONTEXT}

Previous Memory:
{PREV_SUMMARY}

Update the memory using the new frames.

Rules:
- Output ONLY the updated memory and a candidate answer.
- Focus on details relevant to the Question and Options above. Prioritize information that helps answer the question.
- If you observe information that directly answers the question, note it explicitly in [New Change].
- Do not use decorative adjectives or adverbs, such as professional, perfect, clearly, beautiful.
- Keep adjectives only for factual attributes: color, number, position, size, visible state.
- Keep only visible facts useful for future QA.
- Merge previous memory with new frames. Remove low-value details unrelated to the question.
- Preserve important temporal order and state changes, especially those relevant to the question.
- If there is no important change, briefly state the repeated action once.Remove  irrelevant details.
- In [Candidate Answer], give your current best guess from the options based on everything seen so far in this segment and previous memory. If you have insufficient evidence, write "Unknown".

Use this format:
[Scene/Objects] ...
[People/Actions] ...
[Spatial/State] ...
[New Change] ...
[Candidate Answer] <single letter A/B/C/D or "Unknown">
"""

PROMPT_QA_PREFIX = """You are an advanced video question-answering AI assistant.

Previous Video Description:
{PREV_SUMMARY}

Previous Candidate Answers from earlier segments:
{CANDIDATE_ANSWERS}

You have been provided with a text description of previous video content, the candidate answers from earlier video segments, some frames from the current (final) video segment, and a multiple-choice question related to the video. Your task is to carefully analyze the video by combining all of this information, and choose the best answer from the options provided. Treat the previous candidate answers as references from earlier segments, but verify them against the previous memory and current frames before making the final decision. Weight the evidence from segments where the model was confident more heavily than segments marked "Unknown".

"""


def _parse_v2t_response(response: str):
    """Split a V2T response into (memory_body, candidate_answer).

    The model is asked to end its output with a line like:
        [Candidate Answer] A
    We extract that letter/answer and return the rest as the memory body so the
    next chunk's Previous Memory does not carry stale candidate answers (which
    would otherwise anchor the model).
    """
    if not response:
        return "", ""

    marker = "[Candidate Answer]"
    idx = response.rfind(marker)
    if idx == -1:
        return response.strip(), ""

    memory_body = response[:idx].rstrip()
    candidate_part = response[idx + len(marker):].strip()
    # Take the first non-empty token after the marker (the answer itself)
    first_line = candidate_part.split("\n", 1)[0].strip()
    candidate_answer = first_line.split()[0] if first_line else ""
    return memory_body, candidate_answer


def _extract_frames_from_video(video_path, fps=1.0):
    """Extract frames from video at the specified fps using cv2."""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0
    frame_interval = max(1, int(video_fps / fps))
    frames = []
    sampled_indices = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            sampled_indices.append(count)
        count += 1
    cap.release()
    return frames, video_fps, sampled_indices


def _resolve_model_class(pretrained: str, is_moe: bool):
    """Auto-detect and return the appropriate HF model class for a Qwen3 variant.

    Returns (model_class, dtype_kwarg_name) where dtype_kwarg_name is the
    keyword argument name for specifying dtype in from_pretrained().
    """
    config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    if "qwen3_5" in model_type:
        from transformers import (
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForConditionalGeneration,
        )

        model_cls = Qwen3_5MoeForConditionalGeneration if is_moe else Qwen3_5ForConditionalGeneration
        dtype_key = "torch_dtype"
    else:
        from transformers import (
            Qwen3VLForConditionalGeneration,
            Qwen3VLMoeForConditionalGeneration,
        )

        model_cls = Qwen3VLMoeForConditionalGeneration if is_moe else Qwen3VLForConditionalGeneration
        dtype_key = "dtype"

    return model_cls, dtype_key


@register_model("qwen3_vl")
class Qwen3_VL(lmms):
    """
    Unified model class for the Qwen3-VL and Qwen3.5 model families.

    Auto-detects the model variant from the HuggingFace config and loads
    the appropriate model class. Supports both dense and MoE variants.

    - Qwen3-VL: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
    - Qwen3.5:  https://huggingface.co/Qwen/Qwen3.5-4B
    """

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": None,
        "num_beams": 1,
    }

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        total_pixels: Optional[int] = None,
        max_num_frames: int = 32,
        fps: Optional[float] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        enable_thinking: Optional[bool] = None,
        reasoning_prompt: Optional[str] = None,
        # DPMem streaming parameters
        drop_method: Optional[str] = None,
        drop_threshold: Optional[float] = None,
        drop_absolute: Optional[bool] = True,
        dr_save_path: Optional[str] = None,
        frames_per_chunk: int = 20,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # DPMem streaming mode
        self.drop_method = drop_method
        self.drop_threshold = drop_threshold
        self.drop_absolute = drop_absolute
        self.dr_save_path = dr_save_path
        self.frames_per_chunk = int(frames_per_chunk)
        self.streaming_mode = (
            drop_method is not None and str(drop_method).lower() != "none"
        )

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        # When more processes than GPUs, gloo backend is already initialized
        # by __main__.py (NCCL rejects shared GPUs).  Here we just ensure the
        # device mapping uses modulo so multiple ranks share each GPU.
        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            gpu_id = accelerator.local_process_index % torch.cuda.device_count()
            self._device = torch.device(f"cuda:{gpu_id}")
            self.device_map = f"cuda:{gpu_id}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        if self.streaming_mode:
            from lmms_eval.models.simple.modeling_qwen3_vl_dpmem import (
                Qwen3VLForConditionalGeneration as DPMemQwen3VL,
            )
            model_kwargs = {
                "dtype": "bfloat16",
                "device_map": self.device_map,
            }
            if attn_implementation is not None:
                model_kwargs["attn_implementation"] = attn_implementation
            self._model = DPMemQwen3VL.from_pretrained(pretrained, **model_kwargs).eval()
            eval_logger.info(
                f"DPMem streaming mode: drop_method={drop_method}, "
                f"drop_threshold={drop_threshold}, frames_per_chunk={frames_per_chunk}"
            )
        else:
            # Auto-detect model variant and load the appropriate HF class
            is_moe = bool(re.search(r"A\d+B", pretrained))
            model_cls, dtype_key = _resolve_model_class(pretrained, is_moe)

            model_kwargs = {
                dtype_key: "bfloat16",
                "device_map": self.device_map,
            }
            if attn_implementation is not None:
                model_kwargs["attn_implementation"] = attn_implementation

            self._model = model_cls.from_pretrained(pretrained, **model_kwargs).eval()

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.enable_thinking = enable_thinking

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3 VL models")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _build_video_kwargs(self):
        """Build video processing kwargs based on model configuration."""
        video_kwargs = {"min_pixels": self.min_pixels}

        if self.fps is not None:
            video_kwargs["fps"] = self.fps
            video_kwargs["max_frames"] = self.max_num_frames
        elif self.total_pixels is not None:
            video_kwargs["max_frames"] = self.max_num_frames
        else:
            video_kwargs["nframes"] = self.max_num_frames

        if self.total_pixels is not None:
            video_kwargs["total_pixels"] = self.total_pixels
        else:
            video_kwargs["max_pixels"] = self.max_pixels

        return video_kwargs

    def _apply_chat_template(self, batched_messages, **kwargs):
        """Apply chat template with optional enable_thinking support."""
        template_kwargs = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        template_kwargs.update(kwargs)
        return self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True, **template_kwargs)

    def _build_generate_kwargs(self, gen_kwargs):
        """Build model.generate() kwargs from user gen_kwargs merged with defaults."""
        current = {**self.DEFAULT_GEN_KWARGS, **gen_kwargs}
        pad_token_id = self.tokenizer.pad_token_id

        if current.get("temperature", 0) > 0:
            current["do_sample"] = True
        else:
            current["do_sample"] = False
            current["temperature"] = None
            current["top_p"] = None
            current.pop("top_k", None)

        generate_kwargs = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "max_new_tokens": current["max_new_tokens"],
            "use_cache": self.use_cache,
            "do_sample": current["do_sample"],
        }
        for key in ("temperature", "top_p", "top_k", "num_beams"):
            val = current.get(key)
            if val is not None:
                generate_kwargs[key] = val

        return generate_kwargs

    def _strip_thinking(self, answer):
        """Strip <think>...</think> content from model output if enable_thinking is set."""
        if self.enable_thinking:
            _, _, remaining = answer.partition("</think>")
            return remaining.strip()
        return answer

    def _preprocess_chunk(self, chunk):
        """Preprocess a batch chunk on CPU: message building, video decoding, tokenization.

        Returns (inputs, contexts, gen_kwargs, until) with inputs still on CPU.
        """
        contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
        visual_list = [doc_to_visual[0](self.task_dict[t][s][i]) for t, s, i in zip(task, split, doc_id)]
        gen_kwargs = all_gen_kwargs[0]

        until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
        if isinstance(until, str):
            until = [until]
        elif not isinstance(until, list):
            raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
        until = [item for item in until if item != "\n\n"]

        if isinstance(contexts, tuple):
            contexts = list(contexts)

        for i in range(len(contexts)):
            if "<image>" in contexts[i]:
                contexts[i] = contexts[i].replace("<image>", "")

        video_kwargs = self._build_video_kwargs()

        batched_messages = []
        for i, context in enumerate(contexts):
            if "<image>" in context:
                context = context.replace("<image>", "")

            message = [{"role": "system", "content": self.system_prompt}]

            if self.reasoning_prompt:
                context = context.strip() + self.reasoning_prompt
                contexts[i] = context

            processed_visuals = []
            if visual_list[i] is not None:
                for visual in visual_list[i]:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                        processed_visuals.append(
                            {
                                "type": "video",
                                "video": visual,
                                **video_kwargs,
                            }
                        )
                    elif isinstance(visual, Image.Image):
                        processed_visuals.append(
                            {
                                "type": "image",
                                "image": visual,
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )

            if self.interleave_visuals is False:
                message.append(
                    {
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}],
                    }
                )
            else:
                image_placeholders = re.findall(r"<image \d+>", context)
                content_parts = []
                text_parts = re.split(r"<image \d+>", context)
                if text_parts[0]:
                    content_parts.append({"type": "text", "text": text_parts[0]})

                for placeholder_idx, placeholder in enumerate(image_placeholders):
                    img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                    image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                    if processed_visuals and image_idx < len(processed_visuals):
                        content_parts.append(processed_visuals[image_idx])
                    if placeholder_idx + 1 < len(text_parts) and text_parts[placeholder_idx + 1]:
                        content_parts.append({"type": "text", "text": text_parts[placeholder_idx + 1]})

                message.append(
                    {
                        "role": "user",
                        "content": content_parts,
                    }
                )

            batched_messages.append(message)

        texts = self._apply_chat_template(batched_messages)
        image_inputs, video_inputs, processed_video_kwargs = process_vision_info(
            batched_messages,
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )
        video_metadata_list = None
        if video_inputs is not None:
            video_inputs, video_metadata_list = map(list, zip(*video_inputs))

        if self.batch_size > 1:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata_list,
                **processed_video_kwargs,
                do_resize=False,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata_list,
                **processed_video_kwargs,
                do_resize=False,
                return_tensors="pt",
            )

        return inputs, contexts, gen_kwargs, until

    # ------------------------------------------------------------------
    #  Streaming DTD helpers
    # ------------------------------------------------------------------
    def _process_streaming_chunk(
        self, frames_pil, text_content, video_fps, frame_indices, max_new_tokens=256
    ):
        """Run one chunk of PIL frames through the DTD model and return generated text."""
        from transformers.video_utils import VideoMetadata

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames_pil,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": text_content},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        video_meta = VideoMetadata(
            total_num_frames=len(frames_pil),
            fps=video_fps,
            frames_indices=frame_indices if frame_indices else list(range(len(frames_pil))),
        )
        inputs = self.processor(
            text=[text],
            videos=[frames_pil],
            video_metadata=[video_meta],
            do_sample_frames=False,
            padding=True,
            return_tensors="pt",
        )
        if self.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=self.use_cache,
            do_sample=False,
            drop_method=self.drop_method,
            drop_threshold=self.drop_threshold,
            drop_absolute=self.drop_absolute,
            protect_first_last_frames=True,
            dr_save_path=self.dr_save_path,
        )
        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def _streaming_inference(self, video_path, context, gen_kwargs):
        """Process one video with chunk-based streaming DTD compression.

        The question/context is injected into every intermediate V2T prompt so
        that each sliding-window summary is question-aware. Each intermediate
        chunk also emits a candidate answer, which is accumulated and passed to
        the final QA chunk so the model can weigh segment-level evidence rather
        than relying only on the compressed memory.
        """
        sample_fps = self.fps if self.fps else 1.0
        all_frames, video_fps, all_indices = _extract_frames_from_video(
            video_path, sample_fps
        )
        if not all_frames:
            eval_logger.error(f"No frames extracted from {video_path}")
            return ""

        # Clean up the context: strip <image> placeholders and whitespace
        question_context = context.strip() if context else ""

        frame_buf: List[Image.Image] = []
        idx_buf: List[int] = []
        context_text = ""
        candidate_answers: List[str] = []  # per-segment candidate answers
        response = ""

        fi = 0
        total = len(all_frames)
        segment_idx = 0
        while fi < total:
            frame_buf.append(all_frames[fi])
            idx_buf.append(all_indices[fi])
            fi += 1

            is_last = fi >= total
            if len(frame_buf) == self.frames_per_chunk or is_last:
                if is_last:
                    prev_summary = context_text if context_text else " "
                    if candidate_answers:
                        cand_str = "\n".join(
                            f"Segment {i + 1}: {a}"
                            for i, a in enumerate(candidate_answers)
                        )
                    else:
                        cand_str = "None"
                    message_text = (
                        PROMPT_QA_PREFIX.format(
                            PREV_SUMMARY=prev_summary,
                            CANDIDATE_ANSWERS=cand_str,
                        )
                        + question_context
                    )
                    response = self._process_streaming_chunk(
                        frame_buf, message_text, video_fps, idx_buf, 128
                    )
                else:
                    prev_summary = context_text if context_text else " "
                    message_text = PROMPT_V2T.format(
                        QUESTION_CONTEXT=question_context,
                        PREV_SUMMARY=prev_summary,
                    )
                    max_tokens = gen_kwargs.get("max_new_tokens", 256)
                    raw_response = self._process_streaming_chunk(
                        frame_buf, message_text, video_fps, idx_buf, max_tokens
                    )
                    memory_body, cand = _parse_v2t_response(raw_response)
                    context_text = memory_body
                    if cand:
                        candidate_answers.append(cand)
                    else:
                        candidate_answers.append("Unknown")
                    segment_idx += 1
                    frame_buf = [frame_buf[-1]]
                    idx_buf = [idx_buf[-1]]

        return response

    def _generate_until_streaming(self, requests: List[Instance]) -> List[str]:
        """Streaming DTD generate_until: one sample at a time."""
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="DPMem Streaming"
        )
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=1, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            visual_data = doc_to_visual[0](
                self.task_dict[task[0]][split[0]][doc_id[0]]
            )

            gen_kwargs = all_gen_kwargs[0]
            context = contexts[0]
            if "<image>" in context:
                context = context.replace("<image>", "")

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(
                    f"Expected until to be str or list, got {type(until)}"
                )
            until = [u for u in until if u != "\n\n"]

            video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
            if (
                visual_data
                and isinstance(visual_data[0], str)
                and visual_data[0].endswith(video_exts)
            ):
                ans = self._streaming_inference(visual_data[0], context, gen_kwargs)
            else:
                eval_logger.warning(
                    "Non-video input; streaming mode not applicable, returning empty"
                )
                ans = ""

            for term in until:
                if len(term) > 0:
                    ans = ans.split(term)[0]

            res.append(ans)
            self.cache_hook.add_partial(
                "generate_until", (context, gen_kwargs), ans
            )
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    # ------------------------------------------------------------------
    #  Standard (non-streaming) generate_until
    # ------------------------------------------------------------------
    def _generate_until_standard(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = list(re_ords.get_batched(n=self.batch_size, batch_fn=None))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._preprocess_chunk, chunks[0]) if chunks else None

            for idx in range(len(chunks)):
                inputs, contexts, gen_kwargs, until = future.result()

                if idx + 1 < len(chunks):
                    future = executor.submit(self._preprocess_chunk, chunks[idx + 1])

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                generate_kwargs = self._build_generate_kwargs(gen_kwargs)
                cont = self.model.generate(**inputs, **generate_kwargs)

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans

                for ans, context in zip(answers, contexts):
                    ans = self._strip_thinking(ans)
                    res.append(ans)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until(self, requests: List[Instance]) -> List[str]:
        if self.streaming_mode:
            return self._generate_until_streaming(requests)
        return self._generate_until_standard(requests)

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
