"""
NSFW Freepik Nodes for ComfyUI

Detects NSFW content using `nsfw_image_detector` (NSFWDetector).

Features:
- Detects NSFW probabilities per image (uses NSFWLevel.LOW from predict_proba)
- Blocks workflow when LOW probability is greater than or equal to threshold (default 0.5)
"""

import os
import json
from enum import Enum
import torch
from PIL import Image

from nsfw_image_detector import NSFWDetector

import folder_paths
from server import PromptServer
from nodes import interrupt_processing

HF_MODEL_ID = "Freepik/nsfw_image_detector"
LOCAL_MODEL_DIR = os.path.join(folder_paths.models_dir, "nsfw_freepik")


def _model_source() -> str:
    if os.path.isfile(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        return LOCAL_MODEL_DIR
    print(f"[NSFW Freepik] {LOCAL_MODEL_DIR} not found, downloading {HF_MODEL_ID} from Hugging Face")
    return HF_MODEL_ID


class NSFWContentError(Exception):
    """Custom exception for NSFW content detection."""

    def __init__(self, prediction: str, confidence: float, threshold: float):
        self.error_type = "nsfw_content_detected"
        self.prediction = prediction
        self.confidence = confidence
        self.threshold = threshold
        message = (
            "NSFW content detected - workflow interrupted "
            f"(prediction: {prediction}, confidence: {confidence:.2%}, threshold: {threshold:.2%})"
        )
        super().__init__(message)

    def to_dict(self):
        return {
            "error": {
                "type": self.error_type,
                "message": str(self),
                "details": {
                    "prediction": self.prediction,
                    "confidence": self.confidence,
                    "threshold": self.threshold,
                },
            }
        }


class NSFWCheck:
    """Checks images for NSFW content using NSFWDetector."""

    _detector = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": (
                            "Chặn workflow nếu NSFWLevel.LOW >= giá trị này. "
                            "Mặc định 0.5: low >= 0.5 sẽ bị chặn."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "check_nsfw"
    CATEGORY = "safety"

    def _get_detector(self) -> NSFWDetector:
        if NSFWCheck._detector is not None:
            return NSFWCheck._detector

        model_name = _model_source()

        # Prefer requested config (bfloat16 on cuda). Fall back if unsupported.
        try:
            NSFWCheck._detector = NSFWDetector(model_name=model_name, dtype=torch.bfloat16, device="cuda")
        except Exception as e1:
            print(f"[NSFW Freepik] Failed init cuda+bfloat16, falling back: {e1}")
            try:
                NSFWCheck._detector = NSFWDetector(model_name=model_name, dtype=torch.float16, device="cuda")
            except Exception as e2:
                print(f"[NSFW Freepik] Failed init cuda+float16, falling back to CPU: {e2}")
                NSFWCheck._detector = NSFWDetector(model_name=model_name, dtype=torch.float32, device="cpu")

        return NSFWCheck._detector

    def _normalize_prob_key(self, k) -> str:
        # predict_proba uses enum keys like NSFWLevel.LOW; str(k) is not "low".
        if isinstance(k, Enum):
            return str(k.value)
        return str(k)

    def _floatify_prob_value(self, v):
        try:
            if hasattr(v, "item") and callable(getattr(v, "item", None)):
                return float(v.item())
            return float(v)
        except Exception:
            return None

    def _coerce_probabilities(self, probs):
        # Normalize outputs from NSFWDetector.predict_proba into dict[str, float|None].
        def dict_to_out(d: dict) -> dict:
            out = {}
            for k, v in d.items():
                nk = self._normalize_prob_key(k)
                out[nk] = self._floatify_prob_value(v)
            return out

        if isinstance(probs, dict):
            return dict_to_out(probs)

        if isinstance(probs, (list, tuple)) and probs:
            for item in probs:
                if isinstance(item, dict):
                    return dict_to_out(item)

        return probs

    def _extract_low_score(self, probs_dict: dict) -> float:
        # Freepik-style probabilities: high/medium/low/neutral (keys after coercion).
        v = probs_dict.get("low")
        try:
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    def _extract_low_score_any(self, probs_raw) -> float:
        coerced = self._coerce_probabilities(probs_raw)
        if isinstance(coerced, dict):
            return self._extract_low_score(coerced)
        return 0.0

    def check_nsfw(
        self,
        image: torch.Tensor,
        threshold: float = 0.5,
    ):
        detector = self._get_detector()

        batch_size = image.shape[0]

        results = []

        nsfw_detected = False
        max_low_score = 0.0

        for i in range(batch_size):
            # VAE Qwen Image 2.1 trả RGBA (4 kênh); ép 4 kênh thành "RGB" làm detector thấy ảnh nhiễu
            img_tensor = image[i][..., :3].float().clamp(0.0, 1.0)
            img_np = (img_tensor.cpu().numpy() * 255).round().astype("uint8")
            pil_image = Image.fromarray(img_np, mode="RGB")

            probs_raw = detector.predict_proba(pil_image)
            probs = self._coerce_probabilities(probs_raw)

            low_score = self._extract_low_score_any(probs_raw)
            try:
                low_score = float(low_score)
            except Exception:
                low_score = 0.0

            # Optional boolean classification (kept for reference in results)
            try:
                is_nsfw_flag = bool(detector.is_nsfw(pil_image, "low"))
            except Exception as e:
                print(f"[NSFW Freepik] is_nsfw failed for i{i}: {e}")
                is_nsfw_flag = False

            if low_score > max_low_score:
                max_low_score = low_score

            # blocked khi low >= threshold (vd threshold=0.5 → low >= 0.5)
            blocked = low_score >= threshold
            if blocked:
                nsfw_detected = True

            results.append(
                {
                    "index": i,
                    "low_score": low_score,
                    "threshold": threshold,
                    "blocked": blocked,
                    "is_nsfw_flag": is_nsfw_flag,
                    "probabilities": probs,
                }
            )

        if nsfw_detected:
            error = NSFWContentError(
                prediction="nsfw",
                confidence=max_low_score,
                threshold=threshold,
            )

            PromptServer.instance.send_sync(
                "nsfw_guard.content_blocked",
                {
                    "type": "nsfw_content_detected",
                    "prediction": "nsfw",
                    "confidence": max_low_score,
                    "threshold": threshold,
                    "results": results,
                    "message": str(error),
                },
            )

            interrupt_processing(True)
            raise Exception(json.dumps(error.to_dict()))

        return (image,)


NODE_CLASS_MAPPINGS = {
    "NSFWCheck": NSFWCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NSFWCheck": "NSFW Check (Freepik)",
}
