from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ring_console.invalid_zones import make_invalid_zone_mask


RING_DIAMETERS_M = {
    1: 1000.0,
    2: 650.0,
    3: 400.0,
    4: 200.0,
    5: 100.0,
    6: 0.05,
}

DEBUG_DIR = Path(__file__).resolve().parents[1] / "work"


@dataclass
class CropInfo:
    x: int
    y: int
    width: int
    height: int
    mode: str


@dataclass
class RingCircle:
    number: int
    x: float
    y: float
    radius: float
    score: float = 0.0
    source: str = "detected"
    gap: float | None = None

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def with_center(self, center: np.ndarray) -> "RingCircle":
        return RingCircle(
            number=self.number,
            x=float(center[0]),
            y=float(center[1]),
            radius=self.radius,
            score=self.score,
            source=self.source,
            gap=self.gap,
        )

    def to_json(self, crop: CropInfo | None = None) -> dict:
        ox = crop.x if crop else 0
        oy = crop.y if crop else 0
        data = {
            "number": self.number,
            "x": round(self.x + ox, 2),
            "y": round(self.y + oy, 2),
            "cropX": round(self.x, 2),
            "cropY": round(self.y, 2),
            "radius": round(self.radius, 2),
            "score": round(self.score, 4),
            "source": self.source,
        }
        if self.gap is not None:
            data["gap"] = round(self.gap, 2)
        return data


@dataclass
class CandidateCircle:
    x: float
    y: float
    radius: float
    score: float

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


def analyze_image_bytes(
    image_bytes: bytes,
    first_ring: int = 1,
    stage: str = "predict",
    map_id: str | None = None,
    show_invalid_mask: bool = False,
) -> dict:
    started_at = time.perf_counter()
    image = decode_image(image_bytes)
    crop, crop_info = crop_map(image)
    save_debug_image("last_crop.png", crop)
    playable_mask, mask_info = make_invalid_zone_mask(crop.shape[:2], map_id)
    visible_invalid_mask = show_invalid_mask and bool(mask_info.get("enabled"))

    if stage == "crop":
        crop_overlay = apply_invalid_zone_overlay(crop, playable_mask, visible_invalid_mask)
        return {
            "ok": True,
            "stage": "crop",
            "crop": crop_info.__dict__,
            "mask": mask_info,
            "candidateCount": 0,
            "elapsedMs": round((time.perf_counter() - started_at) * 1000),
            "rings": [],
            "predictions": [],
            "candidates": [],
            "overlay": encode_png_data_url(crop_overlay),
        }

    ring_mask, edge_mask = make_ring_masks(crop)
    save_debug_image("last_ring_mask.png", ring_mask)
    save_debug_image("last_edge_mask.png", edge_mask)

    candidates: list[CandidateCircle] = []
    detected = detect_split_hough_pair(crop, first_ring, ring_mask, edge_mask)
    if len(detected) != 2:
        candidates = detect_circle_candidates(crop, ring_mask, edge_mask)
        detected = choose_ring_pair(candidates, first_ring, crop, ring_mask, edge_mask)

    if len(detected) != 2:
        overlay = draw_overlay(
            crop,
            CropInfo(0, 0, crop.shape[1], crop.shape[0], crop_info.mode),
            [],
            [],
            [],
            playable_mask,
            visible_invalid_mask,
        )
        save_debug_image("last_overlay.png", overlay)
        return {
            "ok": False,
            "stage": stage,
            "error": "Could not confidently detect two rings from the cropped map.",
            "crop": crop_info.__dict__,
            "mask": mask_info,
            "candidateCount": len(candidates),
            "elapsedMs": round((time.perf_counter() - started_at) * 1000),
            "rings": [],
            "predictions": [],
            "candidates": [],
            "overlay": encode_png_data_url(overlay),
        }

    detected_overlay = draw_overlay(
        crop,
        CropInfo(0, 0, crop.shape[1], crop.shape[0], crop_info.mode),
        detected,
        [],
        [],
        playable_mask,
        visible_invalid_mask,
    )

    if stage == "detect":
        save_debug_image("last_overlay.png", detected_overlay)
        return {
            "ok": True,
            "stage": "detect",
            "crop": crop_info.__dict__,
            "mask": mask_info,
            "candidateCount": len(candidates),
            "elapsedMs": round((time.perf_counter() - started_at) * 1000),
            "rings": [ring.to_json() for ring in detected],
            "predictions": [],
            "candidates": [],
            "overlay": encode_png_data_url(detected_overlay),
        }

    predictions, prediction_candidates = predict_sequence(
        detected,
        crop_info,
        playable_mask,
        bool(mask_info.get("enabled")),
    )
    overlay = draw_overlay(
        crop,
        CropInfo(0, 0, crop.shape[1], crop.shape[0], crop_info.mode),
        detected,
        predictions,
        prediction_candidates,
        playable_mask,
        visible_invalid_mask,
    )
    save_debug_image("last_overlay.png", overlay)

    return {
        "ok": True,
        "stage": "predict",
        "crop": crop_info.__dict__,
        "mask": mask_info,
        "candidateCount": len(candidates),
        "elapsedMs": round((time.perf_counter() - started_at) * 1000),
        "rings": [ring.to_json() for ring in detected],
        "predictions": [ring.to_json() for ring in predictions],
        "candidates": [ring.to_json() for ring in prediction_candidates],
        "overlay": encode_png_data_url(overlay),
    }


def decode_image(image_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file was not a readable image.")
    return image


def save_debug_image(name: str, image: np.ndarray) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DEBUG_DIR / name), image)


def crop_map(image: np.ndarray) -> tuple[np.ndarray, CropInfo]:
    h, w = image.shape[:2]
    bottom = detect_bottom_hud(image)
    detected = detect_map_bounds(image, bottom)

    if detected is not None:
        x, y, size = detected
        crop = image[y : y + size, x : x + size].copy()
        return crop, CropInfo(x, y, size, size, "detected-map")

    size = int(min(w, bottom))
    x = max(0, (w - size) // 2)
    y = 0
    crop = image[y : y + size, x : x + size].copy()
    return crop, CropInfo(x, y, size, size, "fallback-center-map")


def detect_bottom_hud(image: np.ndarray) -> int:
    h = image.shape[0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row_mean = gray.mean(axis=1)

    min_y = int(h * 0.55)
    for y in range(h - 1, min_y, -1):
        if row_mean[y] > 35:
            return min(h, y + 1)

    return h


def detect_map_bounds(image: np.ndarray, bottom: int) -> tuple[int, int, int] | None:
    h, w = image.shape[:2]
    top = 0
    expected_size = min(w, bottom - top)
    if expected_size < 300:
        return None

    work = image[top:bottom, :, :]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    vertical_diff = np.abs(np.diff(gray.astype(np.int16), axis=1)).mean(axis=0)
    vertical_diff = smooth_signal(vertical_diff, max(9, w // 220))

    expected_left = int(round((w - expected_size) / 2))
    expected_right = expected_left + expected_size
    search_pad = max(70, int(w * 0.08))

    left = strongest_boundary_near(vertical_diff, expected_left, search_pad)
    right = strongest_boundary_near(vertical_diff, expected_right, search_pad)
    if left is None or right is None:
        return None

    right += 1
    detected_size = right - left
    size_error = abs(detected_size - expected_size) / max(expected_size, 1)
    if size_error > 0.055:
        return None

    size = int(round((detected_size + expected_size) / 2))
    x = int(round((left + right - size) / 2))
    y = top

    if x < 0 or y < 0 or x + size > w or y + size > h:
        return None

    return x, y, size


def smooth_signal(signal: np.ndarray, window_size: int) -> np.ndarray:
    if window_size < 3:
        return signal

    if window_size % 2 == 0:
        window_size += 1

    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(signal, kernel, mode="same")


def strongest_boundary_near(signal: np.ndarray, expected: int, pad: int) -> int | None:
    start = max(0, expected - pad)
    end = min(len(signal), expected + pad)
    if end <= start:
        return None

    local = signal[start:end]
    if local.size == 0:
        return None

    threshold = float(np.percentile(signal, 78))
    index = int(np.argmax(local)) + start
    if signal[index] < threshold:
        return None

    return index


def make_ring_masks(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    white = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([179, 95, 255]))
    cyan = cv2.inRange(hsv, np.array([78, 25, 75]), np.array([108, 255, 255]))
    blue = cv2.inRange(hsv, np.array([102, 25, 75]), np.array([130, 255, 255]))

    ring_mask = cv2.bitwise_or(white, cyan)
    ring_mask = cv2.bitwise_or(ring_mask, blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, kernel)
    ring_mask = cv2.dilate(ring_mask, kernel, iterations=1)

    masked_gray = cv2.bitwise_and(gray, gray, mask=ring_mask)
    edges = cv2.Canny(gray, 70, 180)
    ring_edges = cv2.Canny(masked_gray, 35, 125)
    edge_mask = cv2.bitwise_or(edges, ring_edges)

    return ring_mask, edge_mask


def detect_circle_candidates(
    crop: np.ndarray,
    ring_mask: np.ndarray,
    edge_mask: np.ndarray,
) -> list[CandidateCircle]:
    detect_crop, scale = resize_for_detection(crop)
    detect_ring_mask = resize_mask_for_detection(ring_mask, scale)
    detect_edge_mask = resize_mask_for_detection(edge_mask, scale)

    size = min(detect_crop.shape[:2])
    gray = cv2.cvtColor(detect_crop, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.addWeighted(gray, 0.35, detect_ring_mask, 0.65, 0)

    channels = [
        ("enhanced", cv2.GaussianBlur(enhanced, (7, 7), 1.6)),
        ("edges", cv2.GaussianBlur(detect_edge_mask, (5, 5), 1.0)),
    ]

    raw: list[CandidateCircle] = []
    min_radius = max(24, int(size * 0.09))
    max_radius = int(size * 0.64)
    min_dist = max(64, int(size * 0.14))

    for _, channel in channels:
        for param2 in (38, 30, 24):
            found = cv2.HoughCircles(
                channel,
                cv2.HOUGH_GRADIENT,
                dp=1.15,
                minDist=min_dist,
                param1=135,
                param2=param2,
                minRadius=min_radius,
                maxRadius=max_radius,
            )
            if found is None:
                continue

            for x, y, radius in np.round(found[0][:80]).astype(int):
                if not circle_intersects_crop(x, y, radius, detect_crop.shape[:2]):
                    continue
                score = score_circle(detect_ring_mask, detect_edge_mask, x, y, radius)
                if score >= 0.055:
                    raw.append(
                        CandidateCircle(
                            float(x / scale),
                            float(y / scale),
                            float(radius / scale),
                            float(score),
                        )
                    )

    return merge_candidates(raw)


def resize_for_detection(crop: np.ndarray) -> tuple[np.ndarray, float]:
    max_dimension = max(crop.shape[:2])
    target_max = 680
    if max_dimension <= target_max:
        return crop, 1.0

    scale = target_max / max_dimension
    width = max(1, int(round(crop.shape[1] * scale)))
    height = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    return resized, scale


def resize_mask_for_detection(mask: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return mask

    width = max(1, int(round(mask.shape[1] * scale)))
    height = max(1, int(round(mask.shape[0] * scale)))
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def circle_intersects_crop(x: int, y: int, radius: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    return x + radius > 0 and y + radius > 0 and x - radius < w and y - radius < h


def circle_within_crop_margin(
    x: float,
    y: float,
    radius: float,
    shape: tuple[int, int],
    margin_fraction: float,
) -> bool:
    h, w = shape
    margin = min(h, w) * margin_fraction
    return (
        x - radius >= -margin
        and y - radius >= -margin
        and x + radius <= w + margin
        and y + radius <= h + margin
    )


def score_circle(ring_mask: np.ndarray, edge_mask: np.ndarray, x: int, y: int, radius: int) -> float:
    h, w = ring_mask.shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    thickness = max(4, int(min(h, w) / 180))
    cv2.circle(canvas, (x, y), radius, 255, thickness)

    perimeter = np.count_nonzero(canvas)
    if perimeter == 0:
        return 0.0

    ring_hits = np.count_nonzero((ring_mask > 0) & (canvas > 0)) / perimeter
    edge_hits = np.count_nonzero((edge_mask > 0) & (canvas > 0)) / perimeter
    return ring_hits * 0.72 + edge_hits * 0.28


def merge_candidates(candidates: Iterable[CandidateCircle], limit: int = 30) -> list[CandidateCircle]:
    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    kept: list[CandidateCircle] = []

    for candidate in ordered:
        duplicate = False
        for existing in kept:
            center_delta = np.linalg.norm(candidate.center - existing.center)
            radius_delta = abs(candidate.radius - existing.radius)
            if center_delta < 28 and radius_delta < 28:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept[:limit]


def detect_split_hough_pair(
    crop: np.ndarray,
    first_ring: int,
    ring_mask: np.ndarray,
    edge_mask: np.ndarray,
) -> list[RingCircle]:
    if first_ring != 1:
        return []

    size = min(crop.shape[:2])
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1)

    small_candidates = hough_circle_candidates(
        blurred,
        ring_mask,
        edge_mask,
        min_radius=max(35, int(size * 0.045)),
        max_radius=max(90, int(size * 0.24)),
        param2_values=(30, 26, 22),
        limit=60,
    )
    large_candidates = hough_circle_candidates(
        blurred,
        ring_mask,
        edge_mask,
        min_radius=max(150, int(size * 0.20)),
        max_radius=max(260, int(size * 0.36)),
        param2_values=(32, 28, 24),
        limit=80,
    )

    if not small_candidates or not large_candidates:
        return []

    best_score = -1.0
    best_pair: tuple[RingCircle, RingCircle] | None = None

    for large in large_candidates:
        lx, ly, lr = int(round(large.x)), int(round(large.y)), int(round(large.radius))
        large_score = large.score
        if large_score < 0.095:
            continue
        if ly < size * 0.08 or ly > size * 0.92:
            continue
        if not circle_within_crop_margin(lx, ly, lr, crop.shape[:2], 0.04):
            continue
        boundary_score = current_ring_boundary_score(crop, lx, ly, lr)

        for small in small_candidates:
            sx, sy, sr = int(round(small.x)), int(round(small.y)), int(round(small.radius))
            if sr >= lr:
                continue
            if not circle_within_crop_margin(sx, sy, sr, crop.shape[:2], 0.04):
                continue

            ratio = sr / max(lr, 1)
            if ratio < 0.42 or ratio > 0.78:
                continue

            distance = float(np.linalg.norm([sx - lx, sy - ly]))
            gap = float(lr - (distance + sr))
            if gap < -max(8.0, lr * 0.04):
                continue

            small_score = small.score
            gap_score = min(1.0, max(0.0, gap) / max(lr * 0.18, 1.0))
            ratio_score = max(0.0, 1.0 - abs(ratio - 0.58) * 2.7)
            tangent_penalty = 0.18 if gap < -2 else 0.0
            pair_score = (
                large_score * 2.0
                + small_score * 1.2
                + boundary_score * 0.42
                + gap_score * 0.10
                + ratio_score * 0.35
                - tangent_penalty
            )

            if pair_score > best_score:
                best_score = pair_score
                best_pair = (
                    RingCircle(1, float(lx), float(ly), float(lr), float(large_score), "split-hough-large"),
                    RingCircle(2, float(sx), float(sy), float(sr), float(small_score), "split-hough-small", gap),
                )

    if best_pair is None:
        return []

    return [best_pair[0], best_pair[1]]


def current_ring_boundary_score(crop: np.ndarray, x: int, y: int, radius: int) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

    annulus = np.abs(distance - radius) <= 5.0
    inner = np.abs(distance - max(radius - 8, 1)) <= 4.0
    outer = np.abs(distance - (radius + 8)) <= 4.0
    if not np.any(annulus) or not np.any(inner) or not np.any(outer):
        return 0.0

    dark_fraction = float(np.count_nonzero((gray < 90) & annulus) / np.count_nonzero(annulus))
    edge_fraction = float(np.count_nonzero((edges > 0) & annulus) / np.count_nonzero(annulus))
    dark_edge_fraction = float(
        np.count_nonzero((gray < 95) & (edges > 0) & annulus) / np.count_nonzero(annulus)
    )

    red = crop[:, :, 2]
    saturation = hsv[:, :, 1]
    gray_contrast = abs(float(gray[inner].mean()) - float(gray[outer].mean()))
    red_contrast = abs(float(red[inner].mean()) - float(red[outer].mean()))
    saturation_contrast = abs(float(saturation[inner].mean()) - float(saturation[outer].mean()))
    contrast_score = min(1.0, (gray_contrast / 28.0 + red_contrast / 50.0 + saturation_contrast / 55.0) / 3.0)

    return (
        dark_edge_fraction * 1.35
        + edge_fraction * 0.45
        + dark_fraction * 0.18
        + contrast_score * 0.55
    )


def hough_circle_candidates(
    image: np.ndarray,
    ring_mask: np.ndarray,
    edge_mask: np.ndarray,
    min_radius: int,
    max_radius: int,
    param2_values: tuple[int, ...],
    limit: int,
) -> list[CandidateCircle]:
    raw: list[CandidateCircle] = []
    min_dist = max(80, int(min(image.shape[:2]) * 0.08))

    for param2 in param2_values:
        found = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=50,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if found is None:
            continue

        for x, y, radius in np.round(found[0][:80]).astype(int):
            score = score_circle(ring_mask, edge_mask, x, y, radius)
            if score < 0.055:
                continue
            raw.append(CandidateCircle(float(x), float(y), float(radius), float(score)))

    return merge_candidates(raw, limit=limit)


def choose_ring_pair(
    candidates: list[CandidateCircle],
    first_ring: int,
    crop: np.ndarray,
    ring_mask: np.ndarray,
    edge_mask: np.ndarray,
) -> list[RingCircle]:
    if len(candidates) < 2 or first_ring not in RING_DIAMETERS_M:
        return []

    expected_ratio = RING_DIAMETERS_M[first_ring + 1] / RING_DIAMETERS_M[first_ring]
    best_score = -1.0
    best_pair: tuple[CandidateCircle, CandidateCircle, np.ndarray, float, float] | None = None
    outer_candidates = candidates[:8]
    inner_candidates = candidates[:18]
    max_outer_radius = min(crop.shape[:2]) * 0.36

    for outer in outer_candidates:
        if first_ring == 1 and outer.radius > max_outer_radius:
            continue
        if not circle_within_crop_margin(
            outer.x,
            outer.y,
            outer.radius,
            crop.shape[:2],
            0.04,
        ):
            continue

        preferred_gap = max(12.0, outer.radius * 0.045)
        for inner in inner_candidates:
            if outer is inner or outer.radius <= inner.radius:
                continue
            if not circle_within_crop_margin(
                inner.x,
                inner.y,
                inner.radius,
                crop.shape[:2],
                0.04,
            ):
                continue

            ratio = inner.radius / outer.radius
            if abs(ratio - expected_ratio) > 0.30:
                continue

            expected_inner_radius = outer.radius * expected_ratio
            scanned_center, scanned_center_score = fit_scanned_ring_center(crop, outer, expected_inner_radius)
            inner_center = clamp_child_center(
                inner.center,
                outer.center,
                outer.radius,
                expected_inner_radius,
                preferred_gap,
            )
            if scanned_center is not None and scanned_center_score > 0.0:
                scanned_center = clamp_child_center(
                    scanned_center,
                    outer.center,
                    outer.radius,
                    expected_inner_radius,
                    preferred_gap,
                )
                scanned_edge_score = score_circle(
                    ring_mask,
                    edge_mask,
                    int(round(scanned_center[0])),
                    int(round(scanned_center[1])),
                    int(round(expected_inner_radius)),
                )
                if scanned_edge_score + scanned_center_score * 0.35 >= adjusted_center_score(
                    ring_mask,
                    edge_mask,
                    inner_center,
                    expected_inner_radius,
                ):
                    inner_center = scanned_center

            adjusted_inner_score = score_circle(
                ring_mask,
                edge_mask,
                int(round(inner_center[0])),
                int(round(inner_center[1])),
                int(round(expected_inner_radius)),
            )
            center_dist = np.linalg.norm(inner.center - outer.center)
            adjusted_center_dist = np.linalg.norm(inner_center - outer.center)
            containment_gap = outer.radius - (adjusted_center_dist + expected_inner_radius)
            containment_score = min(1.0, max(-1.0, containment_gap / max(preferred_gap, 1.0)))
            ratio_score = max(0.0, 1.0 - abs(ratio - expected_ratio) * 4.0)
            center_adjust_penalty = min(0.7, np.linalg.norm(inner.center - inner_center) / max(outer.radius, 1.0))
            line_score = outer.score + adjusted_inner_score * 1.4
            pair_score = line_score + ratio_score + containment_score - center_adjust_penalty

            if pair_score > best_score:
                best_score = pair_score
                best_pair = (outer, inner, inner_center, expected_inner_radius, adjusted_inner_score)

    if best_pair is None:
        return []

    outer, inner, inner_center, expected_inner_radius, adjusted_inner_score = best_pair
    refined_center, refined_score = refine_circle_center(
        ring_mask,
        edge_mask,
        inner_center,
        expected_inner_radius,
        outer.center,
        outer.radius,
        preferred_gap,
    )
    final_gap = outer.radius - (
        float(np.linalg.norm(refined_center - outer.center)) + float(expected_inner_radius)
    )

    return [
        RingCircle(first_ring, outer.x, outer.y, outer.radius, outer.score, "detected"),
        RingCircle(
            first_ring + 1,
            float(refined_center[0]),
            float(refined_center[1]),
            float(expected_inner_radius),
            max(float(refined_score), float(adjusted_inner_score), float(inner.score)),
            "detected:ratio-locked",
            float(final_gap),
        ),
    ]


def adjusted_center_score(
    ring_mask: np.ndarray,
    edge_mask: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> float:
    return score_circle(
        ring_mask,
        edge_mask,
        int(round(center[0])),
        int(round(center[1])),
        int(round(radius)),
    )


def fit_scanned_ring_center(
    crop: np.ndarray,
    parent: CandidateCircle,
    expected_radius: float,
) -> tuple[np.ndarray | None, float]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([78, 20, 50]), np.array([132, 255, 255]))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_mask)
    best: tuple[float, np.ndarray] | None = None

    for label in range(1, n_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 1200:
            continue

        x = float(stats[label, cv2.CC_STAT_LEFT])
        y = float(stats[label, cv2.CC_STAT_TOP])
        width = float(stats[label, cv2.CC_STAT_WIDTH])
        height = float(stats[label, cv2.CC_STAT_HEIGHT])

        if width < expected_radius * 0.25 or height < expected_radius * 0.2:
            continue

        component_center = np.array(centroids[label], dtype=np.float32)
        distance_to_parent = float(np.linalg.norm(component_center - parent.center))
        if distance_to_parent > parent.radius + expected_radius:
            continue

        component = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        points = np.vstack([contour.reshape(-1, 2) for contour in contours])
        if len(points) < 20:
            continue

        fit_center = least_squares_circle_center(points)
        if fit_center is None:
            continue

        # The scanned-zone fill often appears as a bottom crescent. Its fitted
        # center is noisy, but it correctly indicates the pull direction.
        fit_center = fit_center.astype(np.float32)
        score = area / max(expected_radius * expected_radius, 1.0)
        score += max(0.0, 1.0 - abs(width - expected_radius * 2.0) / max(expected_radius * 2.0, 1.0))
        score += max(0.0, 1.0 - abs((y + height) - (fit_center[1] + expected_radius)) / max(expected_radius, 1.0))

        if best is None or score > best[0]:
            best = (float(score), fit_center)

    if best is None:
        return None, 0.0

    return best[1], best[0]


def least_squares_circle_center(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None

    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    a = np.c_[2.0 * x, 2.0 * y, np.ones(len(points))]
    b = x * x + y * y

    try:
        solution = np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    return np.array([solution[0], solution[1]], dtype=np.float32)


def clamp_child_center(
    child_center: np.ndarray,
    parent_center: np.ndarray,
    parent_radius: float,
    child_radius: float,
    min_gap: float = 0.0,
) -> np.ndarray:
    max_distance = max(0.0, parent_radius - child_radius - min_gap)
    vector = child_center - parent_center
    distance = float(np.linalg.norm(vector))

    if distance <= max_distance:
        return child_center.astype(np.float32)

    if distance == 0.0:
        return parent_center.astype(np.float32)

    return (parent_center + (vector / distance) * max_distance).astype(np.float32)


def refine_circle_center(
    ring_mask: np.ndarray,
    edge_mask: np.ndarray,
    center: np.ndarray,
    radius: float,
    parent_center: np.ndarray,
    parent_radius: float,
    min_gap: float = 0.0,
) -> tuple[np.ndarray, float]:
    best_center = center.astype(np.float32)
    best_score = score_circle(
        ring_mask,
        edge_mask,
        int(round(best_center[0])),
        int(round(best_center[1])),
        int(round(radius)),
    )

    for step in (12, 5, 2):
        improved = True
        while improved:
            improved = False
            for dy in (-step, 0, step):
                for dx in (-step, 0, step):
                    if dx == 0 and dy == 0:
                        continue

                    candidate = best_center + np.array([dx, dy], dtype=np.float32)
                    candidate = clamp_child_center(
                        candidate,
                        parent_center,
                        parent_radius,
                        radius,
                        min_gap,
                    )
                    score = score_circle(
                        ring_mask,
                        edge_mask,
                        int(round(candidate[0])),
                        int(round(candidate[1])),
                        int(round(radius)),
                    )

                    if score > best_score:
                        best_score = score
                        best_center = candidate
                        improved = True

    return best_center, float(best_score)


def predict_sequence(
    detected: list[RingCircle],
    crop: CropInfo,
    playable_mask: np.ndarray,
    mask_active: bool = False,
) -> tuple[list[RingCircle], list[RingCircle]]:
    history = list(detected)
    map_center = np.array([crop.width / 2.0, crop.height / 2.0], dtype=np.float32)

    predictions: list[RingCircle] = []
    first_step_alternatives: list[RingCircle] = []

    while history[-1].number < 5:
        predicted, alternatives = predict_next(history, map_center, playable_mask, mask_active)
        predictions.append(predicted)
        history.append(predicted)

        if not first_step_alternatives:
            first_step_alternatives = alternatives

    return predictions, first_step_alternatives


def predict_next(
    history: list[RingCircle],
    map_center: np.ndarray,
    playable_mask: np.ndarray,
    mask_active: bool = False,
) -> tuple[RingCircle, list[RingCircle]]:
    prev = history[-2]
    current = history[-1]
    next_number = current.number + 1
    next_radius = current.radius * (
        RING_DIAMETERS_M[next_number] / RING_DIAMETERS_M[current.number]
    )

    anchor = map_center if len(history) == 2 else history[-3].center
    pull_vector = current.center - prev.center
    pull_length = float(np.linalg.norm(pull_vector))
    pull_unit, _ = normalized_vector(pull_vector)

    edge_gap = ring_edge_gap(prev, current)
    gap_threshold = max(6.0, prev.radius * 0.025)
    has_gap = edge_gap > gap_threshold
    near_tangent = not has_gap and edge_gap > -max(8.0, prev.radius * 0.04)
    current_max_pull = max(1.0, current.radius - next_radius)
    desired_gap = desired_child_gap(prev, current, next_radius, edge_gap, has_gap)
    desired_pull_distance = max(0.0, current.radius - next_radius - desired_gap)

    wiki_candidates: list[tuple[str, np.ndarray, float]] = []
    closest_edge_add: np.ndarray | None = None
    for vector_name, anchor_vector, vector_prior in wiki_anchor_vectors(anchor, prev, current):
        # The write-up uses V1 +/- V2 from the previous ring center. It also
        # calls out counterpulls, so we keep the opposite subtraction available.
        add_center = prev.center + anchor_vector + pull_vector
        if vector_name == "wiki-closest-edge":
            closest_edge_add = add_center
        wiki_candidates.extend(
            [
                (f"{vector_name}-add", add_center, vector_prior + 0.18),
                (f"{vector_name}-minus-pull", prev.center + anchor_vector - pull_vector, vector_prior - 0.08),
                (f"pull-minus-{vector_name}", prev.center + pull_vector - anchor_vector, vector_prior - 0.24),
            ]
        )

    gap_heading = current.center.copy()
    arc_heading: tuple[str, np.ndarray, float] | None = None
    near_tangent_arc: tuple[str, np.ndarray, float] | None = None
    near_tangent_soft_arc: tuple[str, np.ndarray, float] | None = None
    if pull_length > 0:
        gap_heading = current.center + pull_unit * desired_pull_distance
        if has_gap:
            arc_angle = np.deg2rad(30.0 + 32.0 * abs(float(pull_unit[0])))
            arc_unit = rotate_vector(pull_unit, float(arc_angle))
            arc_heading = (
                "thin-arc",
                current.center + arc_unit * desired_pull_distance,
                2.28,
            )
        elif near_tangent:
            vertical_sign = 1.0 if pull_unit[1] >= 0 else -1.0
            tangent_angle = -vertical_sign * np.deg2rad(58.0)
            tangent_unit = rotate_vector(pull_unit, float(tangent_angle))
            near_tangent_arc = (
                "near-tangent-arc",
                current.center + tangent_unit * (current_max_pull * 0.92),
                1.95,
            )
            near_tangent_soft_arc = (
                "near-tangent-soft-arc",
                current.center + tangent_unit * (current_max_pull * 0.68),
                0.92,
            )

    # If the observed rings have a gap, later zones should prefer the
    # minimum-gap heading without going all the way tangent to the current ring.
    center_scale = 0.25 if has_gap else 0.72
    soft_pull = current.center + pull_vector * center_scale
    closest_edge_heading: tuple[str, np.ndarray, float] | None = None
    if has_gap and closest_edge_add is not None:
        edge_unit, edge_distance = normalized_vector(closest_edge_add - current.center)
        if edge_distance > 0:
            horizontal_pull = abs(float(pull_unit[0]))
            edge_prior = 1.25 + horizontal_pull * 1.35
            closest_edge_heading = (
                "thin-edge",
                current.center + edge_unit * desired_pull_distance,
                edge_prior,
            )

    if has_gap:
        raw_candidates = [
            ("thin-side", gap_heading, 2.12),
            ("soft-pull", soft_pull, 1.16),
            ("center-up", current.center, 0.68),
            ("continue-pull", current.center + pull_vector, 0.28),
        ]
        if arc_heading is not None:
            raw_candidates.insert(1, arc_heading)
        if closest_edge_heading is not None:
            raw_candidates.insert(1, closest_edge_heading)
        raw_candidates.extend((name, center, prior * 0.74) for name, center, prior in wiki_candidates)
    else:
        raw_candidates = [
            ("continue-pull", current.center + pull_vector, 1.12),
            ("soft-pull", soft_pull, 0.62),
            ("center-up", current.center, 0.32),
        ]
        if near_tangent_arc is not None:
            raw_candidates.insert(0, near_tangent_arc)
        if near_tangent_soft_arc is not None:
            raw_candidates.insert(2, near_tangent_soft_arc)
        raw_candidates.extend((name, center, prior) for name, center, prior in wiki_candidates)

    scored: list[tuple[float, RingCircle]] = []
    alternatives: list[RingCircle] = []

    for name, raw_center, prior in raw_candidates:
        candidate_desired_gap = desired_gap
        if has_gap and name == "thin-side":
            max_child_gap = max(0.0, current.radius - next_radius)
            candidate_desired_gap = min(
                max_child_gap * 0.62,
                max(desired_gap, current.radius * 0.15),
            )
        candidate_desired_pull_distance = max(
            0.0,
            current.radius - next_radius - candidate_desired_gap,
        )

        center, clamp_penalty = clamp_inside_parent(
            raw_center,
            current,
            next_radius,
            candidate_desired_gap,
        )
        playable = playable_fraction(center, next_radius, playable_mask)
        center_valid = center_in_mask(center, playable_mask)
        distance_from_current = float(np.linalg.norm(center - current.center))
        child_gap = current.radius - (distance_from_current + next_radius)

        direction_score = 0.0
        if pull_length > 0 and distance_from_current > 0:
            direction_score = float(
                np.dot((center - current.center) / distance_from_current, pull_vector / pull_length)
            )

        if has_gap:
            target_gap_score = max(
                0.0,
                1.0 - abs(child_gap - candidate_desired_gap) / max(candidate_desired_gap, 1.0),
            )
            pull_distance_score = max(
                0.0,
                1.0 - abs(distance_from_current - candidate_desired_pull_distance)
                / max(candidate_desired_pull_distance, 1.0),
            )
            math_score = (
                max(0.0, direction_score) * 0.42
                + target_gap_score * 0.38
                + pull_distance_score * 0.20
            )
        else:
            edge_score = min(1.0, distance_from_current / current_max_pull)
            math_score = edge_score * 0.55 + max(0.0, direction_score) * 0.35

        if "wiki" in name and not has_gap:
            math_score += 0.18

        clamp_weight = 1.55 if has_gap else (0.42 if "wiki" in name else 1.15)
        if mask_active:
            mask_weight = min(1.0, max(0.35, (next_number - 2) / 3.0))
            mask_adjustment = (0.12 if center_valid else -1.25) * mask_weight
            if playable < 0.5:
                mask_adjustment -= (0.5 - playable) * 0.65 * mask_weight
            else:
                mask_adjustment += (playable - 0.5) * 0.10 * mask_weight
            score = prior + 0.16 + math_score + mask_adjustment - clamp_penalty * clamp_weight
        else:
            score = prior + playable * 0.25 + math_score - clamp_penalty * clamp_weight
        circle = RingCircle(
            next_number,
            float(center[0]),
            float(center[1]),
            float(next_radius),
            float(score),
            name,
            float(child_gap),
        )
        alternatives.append(circle)
        scored.append((score, circle))

    scored.sort(key=lambda item: item[0], reverse=True)
    winner = scored[0][1]
    winner.source = f"prediction:{winner.source}"
    display_alternatives = [winner]
    for _, circle in scored[1:]:
        if len(display_alternatives) >= 6:
            break
        if any(
            np.linalg.norm(circle.center - existing.center) < 8.0
            and abs(circle.radius - existing.radius) < 4.0
            for existing in display_alternatives
        ):
            continue
        display_alternatives.append(circle)
    return winner, display_alternatives


def desired_child_gap(
    prev: RingCircle,
    current: RingCircle,
    child_radius: float,
    parent_gap: float,
    has_gap: bool,
) -> float:
    if not has_gap:
        return 0.0

    max_gap = max(0.0, current.radius - child_radius)
    scaled_parent_gap = parent_gap * (current.radius / max(prev.radius, 1.0))
    base_gap = current.radius * 0.09
    desired = max(10.0, base_gap, scaled_parent_gap * 0.50)

    # A gap between R1/R2 means avoid a true edge touch, but the thin side is
    # still the best heading. Keep only a modest buffer so the pull stays clear.
    return min(max_gap * 0.55, desired)


def wiki_anchor_vectors(
    anchor: np.ndarray,
    prev: RingCircle,
    current: RingCircle,
) -> list[tuple[str, np.ndarray, float]]:
    direction, distance_to_prev = normalized_vector(prev.center - anchor)
    if distance_to_prev == 0.0:
        return [("wiki-zero", np.zeros(2, dtype=np.float32), 0.2)]

    vectors: list[tuple[str, np.ndarray, float]] = []
    intersections = ray_circle_intersections(anchor, direction, current.center, current.radius)
    if intersections:
        near = intersections[0]
        vectors.append(("wiki-intersect-near", direction * near, 1.08))
        if len(intersections) > 1:
            vectors.append(("wiki-intersect-far", direction * intersections[-1], 0.58))

    edge_magnitude = abs(prev.radius - distance_to_prev)
    vectors.append(("wiki-closest-edge", direction * edge_magnitude, 1.0))
    vectors.append(("wiki-center", direction * distance_to_prev, 0.42))

    deduped: list[tuple[str, np.ndarray, float]] = []
    for name, vector, prior in vectors:
        if any(np.linalg.norm(vector - kept_vector) < 5.0 for _, kept_vector, _ in deduped):
            continue
        deduped.append((name, vector.astype(np.float32), prior))

    return deduped


def normalized_vector(vector: np.ndarray) -> tuple[np.ndarray, float]:
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        return np.zeros_like(vector, dtype=np.float32), 0.0
    return (vector / length).astype(np.float32), length


def rotate_vector(vector: np.ndarray, radians: float) -> np.ndarray:
    cos_angle = float(np.cos(radians))
    sin_angle = float(np.sin(radians))
    return np.array(
        [
            vector[0] * cos_angle - vector[1] * sin_angle,
            vector[0] * sin_angle + vector[1] * cos_angle,
        ],
        dtype=np.float32,
    )


def ray_circle_intersections(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> list[float]:
    offset = origin - center
    b = 2.0 * float(np.dot(offset, direction))
    c = float(np.dot(offset, offset) - radius**2)
    discriminant = b * b - 4.0 * c
    if discriminant < 0.0:
        return []

    root = float(np.sqrt(discriminant))
    distances = [(-b - root) / 2.0, (-b + root) / 2.0]
    return sorted(distance for distance in distances if distance >= 0.0)


def clamp_inside_parent(
    center: np.ndarray,
    parent: RingCircle,
    child_radius: float,
    min_gap: float = 0.0,
) -> tuple[np.ndarray, float]:
    max_distance = max(0.0, parent.radius - child_radius - min_gap)
    vector = center - parent.center
    distance = float(np.linalg.norm(vector))

    if distance <= max_distance:
        return center.astype(np.float32), 0.0

    if distance == 0.0:
        return parent.center.copy(), 0.0

    clamped = parent.center + (vector / distance) * max_distance
    penalty = min(0.65, (distance - max_distance) / max(parent.radius, 1.0))
    return clamped.astype(np.float32), float(penalty)


def ring_edge_gap(parent: RingCircle, child: RingCircle) -> float:
    return float(parent.radius - (np.linalg.norm(child.center - parent.center) + child.radius))


def playable_fraction(center: np.ndarray, radius: float, playable_mask: np.ndarray) -> float:
    h, w = playable_mask.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    circle = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2

    total = int(np.count_nonzero(circle))
    if total == 0:
        return 0.0

    playable = int(np.count_nonzero(circle & playable_mask))
    return playable / total


def center_in_mask(center: np.ndarray, playable_mask: np.ndarray) -> bool:
    h, w = playable_mask.shape[:2]
    x = int(round(float(center[0])))
    y = int(round(float(center[1])))
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return bool(playable_mask[y, x])


def draw_overlay(
    image: np.ndarray,
    crop: CropInfo,
    detected: list[RingCircle],
    predictions: list[RingCircle],
    alternatives: list[RingCircle],
    playable_mask: np.ndarray | None = None,
    show_invalid_mask: bool = False,
) -> np.ndarray:
    result = apply_invalid_zone_overlay(image, playable_mask, show_invalid_mask)
    overlay = result.copy()

    cv2.rectangle(
        result,
        (crop.x, crop.y),
        (crop.x + crop.width, crop.y + crop.height),
        (40, 190, 255),
        2,
    )

    palette = {
        1: (80, 230, 120),
        2: (255, 180, 70),
        3: (70, 80, 255),
        4: (220, 90, 230),
        5: (50, 220, 245),
    }

    for alt in alternatives:
        if alt.source.startswith("prediction"):
            continue
        draw_circle(overlay, crop, alt, (120, 120, 120), fill_alpha=False, thickness=1)

    for ring in detected:
        draw_circle(overlay, crop, ring, palette.get(ring.number, (255, 255, 255)), False, 4)

    for ring in predictions:
        draw_circle(overlay, crop, ring, palette.get(ring.number, (255, 255, 255)), False, 3)

    cv2.addWeighted(overlay, 0.78, result, 0.22, 0, result)

    all_rings = detected + predictions
    for ring in all_rings:
        color = palette.get(ring.number, (255, 255, 255))
        draw_label(result, crop, ring, color)

    if len(all_rings) >= 2:
        for start, end in zip(all_rings, all_rings[1:]):
            p1 = (int(start.x + crop.x), int(start.y + crop.y))
            p2 = (int(end.x + crop.x), int(end.y + crop.y))
            cv2.line(result, p1, p2, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.circle(result, p2, 4, (245, 245, 245), -1, cv2.LINE_AA)

    return result


def apply_invalid_zone_overlay(
    image: np.ndarray,
    playable_mask: np.ndarray | None,
    enabled: bool,
) -> np.ndarray:
    result = image.copy()
    if not enabled or playable_mask is None:
        return result

    invalid = ~playable_mask.astype(bool)
    if not np.any(invalid):
        return result

    tinted = result.copy()
    tinted[invalid] = (35, 35, 235)
    cv2.addWeighted(tinted, 0.38, result, 0.62, 0, result)

    invalid_u8 = invalid.astype(np.uint8) * 255
    contours, _ = cv2.findContours(invalid_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)
    return result


def draw_circle(
    image: np.ndarray,
    crop: CropInfo,
    ring: RingCircle,
    color: tuple[int, int, int],
    fill_alpha: bool,
    thickness: int,
) -> None:
    center = (int(round(ring.x + crop.x)), int(round(ring.y + crop.y)))
    radius = int(round(ring.radius))

    if fill_alpha:
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)

    cv2.circle(image, center, radius, color, thickness, cv2.LINE_AA)


def draw_label(
    image: np.ndarray,
    crop: CropInfo,
    ring: RingCircle,
    color: tuple[int, int, int],
) -> None:
    label = f"R{ring.number}"
    position = (int(ring.x + crop.x + 10), int(ring.y + crop.y - 10))
    cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)


def encode_png_data_url(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode the result image.")

    payload = base64.b64encode(buffer).decode("ascii")
    return f"data:image/png;base64,{payload}"
