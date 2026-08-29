"""
Qwen3-VL integration for DPMem.

Inherits all base classes from the built-in transformers Qwen3VL implementation,
and adds DPMem visual token compression through subclassing.
"""

import json
import os
import logging
import numpy as np
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as _Qwen3VLForConditionalGeneration,
    Qwen3VLModel as _Qwen3VLModel,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLModelOutputWithPast,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

logger = logging.getLogger(__name__)





class Qwen3VLModel(_Qwen3VLModel):
    """Qwen3VLModel with Dynamic Token Drop (DTD) support."""

    def token_drop(
            self,
        method: str,
        threshold: float,
        absolute: bool = True,
        protect_first_last_frames: bool = False,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list] = None,
        dr_save_path: Optional[str] = None,
        drop_mode: Optional[str] = None,
        is_first_chunk: bool = True,
    ):
        """
        Drop redundant visual tokens based on pixel-level or feature-level
        similarity between consecutive frames. Compared to the default DTD variant,
        redundant positions are removed on the *previous* frame (not the current).

        If protect_first_last_frames is True, tokens on frame 0 are never dropped.
        The last *frame* is never the target of a drop here (drops apply to the
        previous frame only); the final iteration compares last vs penultimate and
        may still drop redundant tokens on the penultimate frame.

        drop_mode controls boundary-frame handling (only effective when
        protect_first_last_frames=True is passed by the caller; the scripts always
        pass True):
            - "both" / None  : mode 3, EXACTLY the current logic. frame 0 protected,
                               frame last naturally protected (never targeted).
            - "first_only"   : mode 1. frame 0 protected, frame last ALSO droppable
                               (extra drop on current frame at the last iteration).
            - "last_only"    : mode 2. frame 0 protected iff is_first_chunk, else
                               droppable. frame last protected.
            - "neither"      : mode 4. frame 0 protected iff is_first_chunk, else
                               droppable. frame last ALSO droppable.

        is_first_chunk only affects "last_only" and "neither": when True, frame 0
        of this chunk is treated as the very first frame of the stream (protected).

        Returns:
            (inputs_embeds, position_ids, visual_pos_masks, deepstack_visual_embeds)
        """
        if absolute is None:
            absolute = True

        if drop_mode is None or drop_mode == "both":
            protect_first = protect_first_last_frames
            drop_last = False
        elif drop_mode == "first_only":
            protect_first = True
            drop_last = True
        elif drop_mode == "last_only":
            protect_first = protect_first_last_frames and bool(is_first_chunk)
            drop_last = False
        elif drop_mode == "neither":
            protect_first = protect_first_last_frames and bool(is_first_chunk)
            drop_last = True
        else:
            raise ValueError(f"Unknown drop_mode: {drop_mode!r}. "
                             f"Expected one of: both, first_only, last_only, neither.")

        batch_size, seq_len, _ = inputs_embeds.size()
        merge_size = 2
        patch_size = 16

        vision_start_token_id = self.config.vision_start_token_id
        vision_end_token_id = self.config.vision_end_token_id

        dropped_hidden_states_list = []
        dropped_pos_ids_list = []
        dropped_visual_pos_masks_list = []

        cnt_total, cnt_drop = 0, 0
        keep_masks_per_batch: List[torch.Tensor] = []

        if not hasattr(self, "_prev_chunk_tail_mask"):
            self._prev_chunk_tail_mask = None
        if is_first_chunk:
            self._prev_chunk_tail_mask = None

        for i in range(batch_size):
            sample_input_ids = input_ids[i]

            vision_start_indices = (sample_input_ids == vision_start_token_id).nonzero(as_tuple=True)[0]
            vision_end_indices = (sample_input_ids == vision_end_token_id).nonzero(as_tuple=True)[0]

            visual_token_indices = []
            for start_idx, end_idx in zip(vision_start_indices, vision_end_indices):
                visual_token_indices.extend(range(start_idx.item() + 1, end_idx.item()))
            visual_token_indices = set(visual_token_indices)

            sample_pos_ids = position_ids[:, i]

            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=inputs_embeds.device)

            if len(visual_token_indices) > 0 and video_grid_thw is not None:
                grid_t, grid_h, grid_w = video_grid_thw[0]
                tokens_per_frame = (grid_h * grid_w) // (merge_size * merge_size)
                sorted_visual_indices = sorted(list(visual_token_indices))

                if len(sorted_visual_indices) != grid_t * tokens_per_frame:
                    logger.warning(
                        f"Token count mismatch! Expected {grid_t}*{tokens_per_frame}="
                        f"{grid_t * tokens_per_frame}, got {len(sorted_visual_indices)}"
                    )
                    vis_frames = np.array_split(sorted_visual_indices, max(1, grid_t.item()))
                    vis_frames = [torch.tensor(x, device=inputs_embeds.device) for x in vis_frames]
                else:
                    vis_frames = []
                    for frame_idx in range(grid_t):
                        start = frame_idx * tokens_per_frame
                        end = start + tokens_per_frame
                        vis_frames.append(
                            torch.tensor(sorted_visual_indices[start:end], device=inputs_embeds.device)
                        )

                if ((not protect_first) and (not is_first_chunk)
                        and self._prev_chunk_tail_mask is not None
                        and len(vis_frames) > 0):
                    first_frame_indices = vis_frames[0]
                    cached_tail = self._prev_chunk_tail_mask.to(keep_mask.device)
                    if cached_tail.numel() == len(first_frame_indices):
                        cnt_drop += int((~cached_tail).sum().item())
                        keep_mask[first_frame_indices] = (
                            keep_mask[first_frame_indices] & cached_tail
                        )

                if method == 'pixel' and pixel_values_videos is not None:
                    channel = 3
                    temporal_patch_size = 2
                    num_tokens_raw = (
                        grid_t * (grid_h // merge_size) * (grid_w // merge_size)
                        * merge_size * merge_size
                    )
                    pixel_values_video_i = pixel_values_videos[:num_tokens_raw]
                    pixel_values_video_i = pixel_values_video_i.view(
                        grid_t, grid_h // merge_size, grid_w // merge_size,
                        merge_size, merge_size, channel, temporal_patch_size,
                        patch_size, patch_size,
                    ).permute(0, 6, 1, 2, 5, 3, 4, 7, 8).reshape(
                        grid_t, temporal_patch_size,
                        (grid_h // merge_size) * (grid_w // merge_size),
                        channel, -1,
                    )

                    prev_pixel_frame = None
                    for frame_idx, frame_indices in enumerate(vis_frames):
                        curr_pixel_frame = pixel_values_video_i[frame_idx, -1]
                        cnt_total += len(frame_indices)

                        if prev_pixel_frame is None:
                            prev_pixel_frame = pixel_values_video_i[frame_idx, 0]
                            continue


                        if protect_first and frame_idx - 1 == 0:
                            prev_pixel_frame = pixel_values_video_i[frame_idx, 0]
                            continue

                        prev_start_idx = vis_frames[frame_idx - 1][0]

                        difference = torch.abs(prev_pixel_frame - curr_pixel_frame).mean(dim=-1).mean(dim=-1)

                        if absolute:
                            drop_local = (difference < threshold).nonzero(as_tuple=True)[0]
                        else:
                            k = int(difference.numel() * threshold)
                            drop_local = torch.topk(difference, k, largest=False).indices

                        drop_indices = drop_local.to(prev_start_idx.device) + prev_start_idx
                        cnt_drop += len(drop_indices)
                        prev_pixel_frame = pixel_values_video_i[frame_idx, 0]
                        keep_mask[drop_indices] = False

                        if drop_last and frame_idx == len(vis_frames) - 1:
                            curr_start_idx = frame_indices[0]
                            drop_indices_curr = drop_local.to(curr_start_idx.device) + curr_start_idx
                            cnt_drop += len(drop_indices_curr)
                            keep_mask[drop_indices_curr] = False

                elif method == 'feature':
                    prev_frame = None
                    for frame_idx, frame_indices in enumerate(vis_frames):
                        curr_frame = inputs_embeds[i, frame_indices]
                        cnt_total += len(frame_indices)

                        if protect_first and frame_idx == 0:
                            prev_frame = curr_frame
                            continue
                        if prev_frame is None:
                            prev_frame = curr_frame
                            continue


                        if protect_first and frame_idx - 1 == 0:
                            prev_frame = curr_frame
                            continue

                        prev_start_idx = vis_frames[frame_idx - 1][0]

                        similarity = F.cosine_similarity(prev_frame, curr_frame, dim=1)

                        if absolute:
                            drop_local_idx = (similarity > threshold).nonzero(as_tuple=True)[0]
                        else:
                            k = int(similarity.numel() * threshold)
                            drop_local_idx = torch.topk(similarity, k).indices

                        drop_indices = drop_local_idx.to(prev_start_idx.device) + prev_start_idx
                        cnt_drop += len(drop_indices)
                        keep_mask[drop_indices] = False

                        if drop_last and frame_idx == len(vis_frames) - 1:
                            curr_start_idx = frame_indices[0]
                            drop_indices_curr = drop_local_idx.to(curr_start_idx.device) + curr_start_idx
                            cnt_drop += len(drop_indices_curr)
                            keep_mask[drop_indices_curr] = False

                        prev_frame = curr_frame


            if (
                i == 0
                and len(visual_token_indices) > 0
                and video_grid_thw is not None
                and len(vis_frames) > 0
            ):
                self._prev_chunk_tail_mask = keep_mask[vis_frames[-1]].detach().clone()

            dropped_hidden = inputs_embeds[i][keep_mask].unsqueeze(0)
            dropped_pos_ids_i = sample_pos_ids[:, keep_mask].unsqueeze(1)

            if visual_pos_masks is not None:
                dropped_vis_mask_i = visual_pos_masks[i][keep_mask].unsqueeze(0)
            else:
                dropped_vis_mask_i = None

            dropped_hidden_states_list.append(dropped_hidden)
            dropped_pos_ids_list.append(dropped_pos_ids_i)
            dropped_visual_pos_masks_list.append(dropped_vis_mask_i)
            keep_masks_per_batch.append(keep_mask)

        dropped_inputs_embeds = torch.cat(dropped_hidden_states_list, dim=0)
        dropped_pos_ids = torch.cat(dropped_pos_ids_list, dim=1)

        if visual_pos_masks is not None:
            dropped_visual_pos_masks = torch.cat(dropped_visual_pos_masks_list, dim=0)
        else:
            dropped_visual_pos_masks = None

        dropped_deepstack = None
        if deepstack_visual_embeds is not None and dropped_visual_pos_masks is not None:
            dropped_deepstack = []
            for layer_emb in deepstack_visual_embeds:
                new_embs = []
                offset = 0
                for i in range(batch_size):
                    orig_vis_mask = visual_pos_masks[i]
                    n_vis = orig_vis_mask.sum().item()
                    vis_keep = keep_masks_per_batch[i][orig_vis_mask]
                    new_embs.append(layer_emb[offset:offset + n_vis][vis_keep])
                    offset += n_vis
                dropped_deepstack.append(torch.cat(new_embs, dim=0))

        if dr_save_path and cnt_total > 0:
            drop_ratio_info = {'drop': cnt_drop, 'total': cnt_total, 'ratio': cnt_drop / cnt_total}
            with open(dr_save_path, 'a' if os.path.exists(dr_save_path) else 'w') as f:
                f.write(json.dumps(drop_ratio_info) + '\n')

        return dropped_inputs_embeds, dropped_pos_ids, dropped_visual_pos_masks, dropped_deepstack

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        drop_method=None,
        drop_threshold=None,
        drop_absolute=None,
        protect_first_last_frames=None,
        dr_save_path=None,
        drop_mode=None,
        is_first_chunk=True,
        **kwargs,
    ):
        """Forward with DTD support. Mirrors the parent but inserts token_drop."""

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_outputs.pooler_output
            deepstack_image_embeds = image_outputs.deepstack_features
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            


        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = video_outputs.pooler_output
            deepstack_video_embeds = video_outputs.deepstack_features
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if self.rope_deltas is None or past_key_values_length == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (past_key_values_length + self.rope_deltas).to(inputs_embeds.device)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if (drop_method is not None and drop_method.lower() != 'none'
                and input_ids is not None
                and self.config.vision_start_token_id in input_ids[0]):
            inputs_embeds, position_ids, visual_pos_masks, deepstack_visual_embeds = self.token_drop(
                method=drop_method,
                threshold=drop_threshold,
                absolute=drop_absolute if drop_absolute is not None else True,
                protect_first_last_frames=protect_first_last_frames or False,
                inputs_embeds=inputs_embeds,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                position_ids=position_ids,
                input_ids=input_ids,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                dr_save_path=dr_save_path,
                drop_mode=drop_mode,
                is_first_chunk=is_first_chunk,
            )
            input_ids = None
            new_seq_len = inputs_embeds.shape[1]
            cache_position = torch.arange(new_seq_len, device=inputs_embeds.device)
            attention_mask = None

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


class Qwen3VLForConditionalGeneration(_Qwen3VLForConditionalGeneration):
    """Qwen3VLForConditionalGeneration with Dynamic Token Drop (DTD) support."""

    def __init__(self, config):
        super().__init__(config)
        self.model.__class__ = Qwen3VLModel

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        logits_to_keep=0,
        drop_method=None,
        drop_threshold=None,
        drop_absolute=None,
        protect_first_last_frames=None,
        dr_save_path=None,
        drop_mode=None,
        is_first_chunk=True,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            drop_method=drop_method,
            drop_threshold=drop_threshold,
            drop_absolute=drop_absolute,
            protect_first_last_frames=protect_first_last_frames,
            dr_save_path=dr_save_path,
            drop_mode=drop_mode,
            is_first_chunk=is_first_chunk,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        drop_method=None,
        drop_threshold=None,
        drop_absolute=None,
        protect_first_last_frames=None,
        dr_save_path=None,
        drop_mode=None,
        is_first_chunk=True,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        model_inputs["drop_method"] = drop_method
        model_inputs["drop_threshold"] = drop_threshold
        model_inputs["drop_absolute"] = drop_absolute
        model_inputs["protect_first_last_frames"] = protect_first_last_frames
        model_inputs["dr_save_path"] = dr_save_path
        model_inputs["drop_mode"] = drop_mode
        model_inputs["is_first_chunk"] = is_first_chunk

        return model_inputs


__all__ = [
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
]
