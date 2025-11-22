import torch
import numpy as np
from PIL import Image

from comfy import model_management as mm

from transformers import (
    Sam3VideoModel,
    Sam3VideoProcessor,
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
)

# -------------------------------
# Global lazy-loaded SAM3 models
# -------------------------------

_sam3_video_model = None
_sam3_video_processor = None
_sam3_tracker_model = None
_sam3_tracker_processor = None


def get_sam3_video(device):
    """Lazy-load SAM3 Video model + processor."""
    global _sam3_video_model, _sam3_video_processor
    if _sam3_video_model is None:
        _sam3_video_model = Sam3VideoModel.from_pretrained("facebook/sam3")
        _sam3_video_model.to(device, dtype=torch.bfloat16)
        _sam3_video_model.eval()
        _sam3_video_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    return _sam3_video_model, _sam3_video_processor


def get_sam3_tracker(device):
    """Lazy-load SAM3 Tracker Video model + processor."""
    global _sam3_tracker_model, _sam3_tracker_processor
    if _sam3_tracker_model is None:
        _sam3_tracker_model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3")
        _sam3_tracker_model.to(device, dtype=torch.bfloat16)
        _sam3_tracker_model.eval()
        _sam3_tracker_processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
    return _sam3_tracker_model, _sam3_tracker_processor


def to_pil_frames(images_tensor: torch.Tensor):
    """Convert ComfyUI IMAGE batch [B,H,W,C] in 0–1 to list[PIL.Image]."""
    imgs = images_tensor.detach().cpu().numpy()
    imgs = (imgs * 255.0).clip(0, 255).astype("uint8")
    frames = [Image.fromarray(imgs[i]) for i in range(imgs.shape[0])]
    return frames


def binary_mask_tensor(masks_np, height: int, width: int):
    """Convert numpy mask(s) from SAM3 into [1,H,W] float tensor in 0–1."""
    if masks_np is None:
        out = np.zeros((height, width), dtype=np.float32)
    else:
        m = masks_np
        if m.ndim == 3:  # [N,H,W] -> union
            m = (m.sum(axis=0) > 0).astype(np.float32)
        elif m.ndim == 2:
            m = m.astype(np.float32)
        else:
            raise ValueError(f"Unexpected mask shape: {m.shape}")
        out = m
    t = torch.from_numpy(out)[None, :, :]  # [1,H,W]
    return t


# -----------------------------------------
# Node: SAM3 – Text → Video Mask
# -----------------------------------------

class SAM3_TextToVideoMask:
    """Use SAM3Video with a text prompt to get a mask per frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"default": "basketball", "multiline": False}),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "run"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = "Create a video mask from a text prompt using SAM3 Video."

    def run(self, images, text, threshold):
        device = mm.get_torch_device()

        frames = to_pil_frames(images)
        num_frames = len(frames)
        if num_frames == 0:
            # Empty batch – return zero mask matching spatial dims
            return (torch.zeros_like(images[:, :, :, 0]),)

        height = frames[0].height
        width = frames[0].width

        video_model, video_processor = get_sam3_video(device)

        video_session = video_processor.init_video_session(
            video=frames,
            inference_device=device,
            dtype=torch.bfloat16,
        )

        masks_per_frame = {}

        for t in range(num_frames):
            batch = video_processor.preprocess_step(
                video_infer_session=video_session,
                frame_idx=t,
                text_queries=[text],
                multimask_output=False,
                target_frame=-1,
            )

            with torch.no_grad():
                outputs = video_model(**batch.to(device, dtype=torch.bfloat16))
                processed = video_processor.postprocess_step(
                    model_outputs=outputs,
                    video_infer_session=video_session,
                    binarize_thresh=threshold,
                )

            masks = processed.get("masks", None)  # [num_objs,H,W] or None
            if masks is None or masks.shape[0] == 0:
                masks_per_frame[t] = None
            else:
                m_bin = (masks > threshold).cpu().numpy().astype("uint8") * 255
                masks_per_frame[t] = m_bin

        mask_batch = []
        for idx in range(num_frames):
            m_np = masks_per_frame.get(idx, None)
            m_tensor = binary_mask_tensor(m_np, height, width)  # [1,H,W]
            mask_batch.append(m_tensor)

        mask_batch = torch.stack(mask_batch, dim=0)  # [B,1,H,W]
        mask_batch = mask_batch[:, 0, :, :]          # [B,H,W]
        mask_batch = mask_batch.clamp(0, 1)

        return (mask_batch,)


# -----------------------------------------
# Node: SAM3 – Point → Video Mask (Tracker)
# -----------------------------------------

class SAM3_PointToVideoMask:
    """Use SAM3TrackerVideo with a single point click to track an object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "x": ("INT", {"default": 320, "min": 0, "max": 8192}),
                "y": ("INT", {"default": 180, "min": 0, "max": 8192}),
                "ann_frame_idx": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "run"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = "Create a video mask from a point click using SAM3 Tracker Video."

    def run(self, images, x, y, ann_frame_idx, threshold):
        device = mm.get_torch_device()

        frames = to_pil_frames(images)
        num_frames = len(frames)
        if num_frames == 0:
            return (torch.zeros_like(images[:, :, :, 0]),)

        ann_frame_idx = int(max(0, min(num_frames - 1, ann_frame_idx)))
        height = frames[0].height
        width = frames[0].width

        tracker_model, tracker_processor = get_sam3_tracker(device)

        tracker_session = tracker_processor.init_video_session(
            video=frames,
            inference_device=device,
            dtype=torch.bfloat16,
        )

        # Point annotation: [images, objects, points_per_object, 2]
        points = [[[[float(x), float(y)]]]]
        obj_ids = [[1]]  # arbitrary object id

        # Step A: annotate on the chosen frame
        batch_anno = tracker_processor.preprocess_annotate_step(
            tracker_infer_session=tracker_session,
            frame_idx=ann_frame_idx,
            points=points,
            labels=[[1]],  # 1 = foreground
            multimask_output=False,
        )

        with torch.no_grad():
            anno_outputs = tracker_model(**batch_anno.to(device, dtype=torch.bfloat16))
            tracker_processor.postprocess_annotate_step(
                model_outputs=anno_outputs,
                tracker_infer_session=tracker_session,
                binarize_thresh=threshold,
            )

        # Step B: track across all frames
        masks_per_frame = {}

        for frame_idx in range(num_frames):
            batch_track = tracker_processor.preprocess_track_step(
                tracker_infer_session=tracker_session,
                frame_idx=frame_idx,
                obj_ids=obj_ids,
            )

            with torch.no_grad():
                track_outputs = tracker_model(**batch_track.to(device, dtype=torch.bfloat16))
                processed = tracker_processor.postprocess_track_step(
                    model_outputs=track_outputs,
                    tracker_infer_session=tracker_session,
                    binarize_thresh=threshold,
                )

            masks = processed.get("masks", None)  # [N,H,W] or None
            if masks is None or masks.shape[0] == 0:
                masks_per_frame[frame_idx] = None
            else:
                m_bin = (masks > threshold).cpu().numpy().astype("uint8") * 255
                masks_per_frame[frame_idx] = m_bin

        mask_batch = []
        for idx in range(num_frames):
            m_np = masks_per_frame.get(idx, None)
            m_tensor = binary_mask_tensor(m_np, height, width)  # [1,H,W]
            mask_batch.append(m_tensor)

        mask_batch = torch.stack(mask_batch, dim=0)  # [B,1,H,W]
        mask_batch = mask_batch[:, 0, :, :]          # [B,H,W]
        mask_batch = mask_batch.clamp(0, 1)

        return (mask_batch,)


# -----------------------------------------
# ComfyUI registration
# -----------------------------------------

NODE_CLASS_MAPPINGS = {
    "SAM3_TextToVideoMask": SAM3_TextToVideoMask,
    "SAM3_PointToVideoMask": SAM3_PointToVideoMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3_TextToVideoMask": "SAM3 – Text → Video Mask",
    "SAM3_PointToVideoMask": "SAM3 – Point → Video Mask",
}
