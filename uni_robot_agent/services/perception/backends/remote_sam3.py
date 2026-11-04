"""Remote SAM 3.1 segmentation over an image/prompt HTTP API."""
from __future__ import annotations

import base64
import binascii
import io
import json
import logging
from collections.abc import Sequence
from typing import Any, ClassVar

import httpx
import numpy as np
from PIL import Image

from ..backend import InvalidImage, InvalidQuery, SegmentationResult, SegmentInput
from ..contracts import PointQuery, TextQuery

logger = logging.getLogger(__name__)

_HEALTH_PATH = "/api/health"
_SEGMENT_PATH = "/api/segment"
_READY_STATES = {"ready", "ok"}
# The remote service returns every candidate above this floor; the harness applies
# min_score itself so below_threshold stays distinguishable from no_match.
_CANDIDATE_FLOOR = 0.0


class RemoteSam3Backend:
    """Delegate inference to a hosted SAM 3.1 service and apply harness selection."""

    name = "remote_sam3"
    capabilities = ("text_segmentation", "point_segmentation")
    batching_capabilities: ClassVar[dict[str, bool]] = {
        "text_prompts": True, "point_prompts": True,
    }

    def __init__(self, endpoint: str, *, timeout_s: float = 450.0) -> None:
        if not endpoint.strip():
            raise ValueError("Remote SAM3 endpoint must be a nonempty URL")
        self._endpoint = endpoint.rstrip("/")
        self._client = httpx.Client(timeout=timeout_s)

    @classmethod
    def connect(cls, endpoint: str, *, timeout_s: float = 450.0) -> RemoteSam3Backend:
        """Construct a backend only when the remote service reports readiness."""
        backend = cls(endpoint, timeout_s=timeout_s)
        try:
            info = backend._health()
        except BaseException:
            backend.close()
            raise
        logger.info(
            "remote SAM3 ready at %s: model=%s device=%s",
            backend._endpoint, info.get("model"), info.get("device"),
        )
        return backend

    def _url(self, path: str) -> str:
        return f"{self._endpoint}{path}"

    def _health(self) -> dict[str, Any]:
        response = self._client.get(self._url(_HEALTH_PATH))
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Remote SAM3 health response is not JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Remote SAM3 health response must be an object")
        status = str(payload.get("status", "")).lower()
        if status not in _READY_STATES:
            raise RuntimeError(
                f"Remote SAM3 is not ready: status={payload.get('status')!r}"
            )
        return payload

    def close(self) -> None:
        self._client.close()

    def process_batch(
        self, jobs: Sequence[SegmentInput],
    ) -> list[SegmentationResult | BaseException]:
        results: list[SegmentationResult | BaseException] = []
        for job in jobs:
            try:
                results.append(self._segment(job))
            except BaseException as exc:  # noqa: BLE001 - isolate one request's failure.
                results.append(exc)
        return results

    def _segment(self, job: SegmentInput) -> SegmentationResult:
        height, width, filename, media_type = self._inspect(job.image_bytes)
        options = {
            "prompts": [self._prompt(job.query, height, width)],
            "threshold": _CANDIDATE_FLOOR,
        }
        try:
            response = self._client.post(
                self._url(_SEGMENT_PATH),
                data={"options": json.dumps(options)},
                files={"file": (filename, job.image_bytes, media_type)},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {400, 413, 415, 422}:
                raise InvalidImage(
                    f"Remote SAM3 rejected the image: HTTP {status}"
                ) from exc
            raise RuntimeError(
                f"Remote SAM3 rejected the request: HTTP {status}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Remote SAM3 request failed: {exc}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Remote SAM3 segment response is not JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Remote SAM3 segment response must be an object")
        return self._select(payload, min_score=job.min_score,
                            image_shape=(height, width))

    @staticmethod
    def _prompt(query: TextQuery | PointQuery, height: int, width: int) -> dict[str, Any]:
        if isinstance(query, TextQuery):
            return {"text": query.text}
        row, col = query.point
        if row < 0 or col < 0 or row >= height or col >= width:
            raise InvalidQuery(
                f"point [row, col] {query.point} is outside image shape "
                f"[{height}, {width}]"
            )
        # The remote API takes pixel [x, y]; the harness query is (row, column).
        return {"points": [[col, row]], "labels": [1]}

    @staticmethod
    def _inspect(image_bytes: bytes) -> tuple[int, int, str, str]:
        """Return image shape plus the upload name and type the remote service requires."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
                format_name = (image.format or "").upper()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise InvalidImage("Could not read input image dimensions") from exc
        if width <= 0 or height <= 0:
            raise InvalidImage("Input image dimensions must be positive")
        # The remote service picks a decoder from the file extension.
        uploads = {
            "PNG": ("image.png", "image/png"),
            "JPEG": ("image.jpg", "image/jpeg"),
            "WEBP": ("image.webp", "image/webp"),
        }
        if format_name not in uploads:
            raise InvalidImage(
                f"Remote SAM3 accepts PNG, JPEG or WebP images, not {format_name or 'unknown'}"
            )
        filename, media_type = uploads[format_name]
        return height, width, filename, media_type

    def _select(
        self, payload: dict[str, Any], *, min_score: float,
        image_shape: tuple[int, int],
    ) -> SegmentationResult:
        candidates = self._candidates(payload, image_shape=image_shape)
        if not candidates:
            return SegmentationResult(status="no_match", reason="Remote SAM3 returned no candidate")
        best = max(candidates, key=lambda item: item["score"])
        score, box = float(best["score"]), best["box"]
        if score < min_score:
            return SegmentationResult(
                status="below_threshold", score=score, box_xyxy=box,
                reason=f"top score {score:.3f} is below min_score {min_score:.3f}",
            )
        mask = self._mask(best["mask"], image_shape=image_shape)
        if not mask.any():
            return SegmentationResult(
                status="empty_mask", score=score, box_xyxy=box,
                reason="Remote SAM3 returned an empty mask",
            )
        return SegmentationResult(status="matched", score=score, box_xyxy=box, mask=mask)

    @staticmethod
    def _candidates(
        payload: dict[str, Any], *, image_shape: tuple[int, int],
    ) -> list[dict[str, Any]]:
        height, width = image_shape
        results = payload.get("results")
        if not isinstance(results, list):
            raise RuntimeError("Remote SAM3 response is missing results")
        candidates: list[dict[str, Any]] = []
        for result in results:
            if not isinstance(result, dict):
                raise RuntimeError("Remote SAM3 result entries must be objects")
            masks = result.get("masks") or []
            scores = result.get("scores") or result.get("out_probs") or []
            boxes = result.get("boxes")
            if boxes is None:
                boxes = [
                    RemoteSam3Backend._scale_box(box, height=height, width=width)
                    for box in (result.get("out_boxes_xywh") or [])
                ]
            if not isinstance(masks, list) or not isinstance(scores, list) or not isinstance(boxes, list):
                raise RuntimeError("Remote SAM3 candidate fields must be lists")
            if not len(masks) == len(scores) == len(boxes):
                raise RuntimeError("Remote SAM3 candidate lists have inconsistent lengths")
            for mask, score, box in zip(masks, scores, boxes, strict=True):
                value = float(score)
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise RuntimeError("Remote SAM3 scores must be finite probabilities")
                if not isinstance(mask, str) or not mask:
                    raise RuntimeError("Remote SAM3 masks must be base64 strings")
                candidates.append({
                    "score": value,
                    "mask": mask,
                    "box": tuple(float(value) for value in box),
                })
        return candidates

    @staticmethod
    def _scale_box(box: Any, *, height: int, width: int) -> tuple[float, float, float, float]:
        cx, cy, box_width, box_height = box
        return (
            (cx - box_width / 2) * width,
            (cy - box_height / 2) * height,
            (cx + box_width / 2) * width,
            (cy + box_height / 2) * height,
        )

    @staticmethod
    def _mask(encoded: str, *, image_shape: tuple[int, int]) -> np.ndarray:
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise RuntimeError("Remote SAM3 mask is not valid base64") from exc
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "PNG" or image.mode not in {"1", "L"}:
                    raise ValueError("mask must be a binary grayscale PNG")
                pixels = np.asarray(image.convert("L"))
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise RuntimeError(f"Could not decode remote segmentation mask: {exc}") from exc
        if pixels.shape != image_shape:
            raise RuntimeError(
                f"Remote SAM3 mask shape {pixels.shape} does not match image {image_shape}"
            )
        return pixels > 127
