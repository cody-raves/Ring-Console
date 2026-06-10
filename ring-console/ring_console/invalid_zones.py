from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data" / "invalid_zones"
DEFAULT_MAP_ID = "mp_rr_olympus_mu3"

MAP_OPTIONS = {
    "mp_rr_olympus_mu3": "Olympus",
    "mp_rr_canyonlands_hu": "Kings Canyon",
    "mp_rr_tropic_island_mu2": "Storm Point",
}


@dataclass(frozen=True)
class InvalidZoneLayer:
    map_id: str
    name: str
    source_url: str
    image_size: int
    invalid_zones: tuple[dict, ...]
    end_locations: tuple[dict, ...]

    def summary(self, enabled: bool = True) -> dict:
        return {
            "enabled": enabled,
            "mapId": self.map_id,
            "name": self.name,
            "imageSize": self.image_size,
            "invalidZoneCount": len(self.invalid_zones),
            "endLocationCount": len(self.end_locations),
            "sourceUrl": self.source_url,
        }


def available_map_options() -> list[dict]:
    return [
        {"id": map_id, "name": name, "default": map_id == DEFAULT_MAP_ID}
        for map_id, name in MAP_OPTIONS.items()
    ]


@lru_cache(maxsize=None)
def load_invalid_zone_layer(map_id: str | None) -> InvalidZoneLayer | None:
    if not map_id or map_id == "none":
        return None

    if map_id not in MAP_OPTIONS:
        return None

    path = DATA_DIR / f"{map_id}.json"
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    return InvalidZoneLayer(
        map_id=payload["mapId"],
        name=payload.get("name", MAP_OPTIONS[map_id]),
        source_url=payload.get("sourceUrl", ""),
        image_size=int(payload.get("imageSize", 16384)),
        invalid_zones=tuple(payload.get("invalidEndZones", [])),
        end_locations=tuple(payload.get("endLocation", [])),
    )


def make_invalid_zone_mask(shape: tuple[int, int], map_id: str | None) -> tuple[np.ndarray, dict]:
    height, width = shape
    mask = np.ones((height, width), dtype=np.uint8)
    layer = load_invalid_zone_layer(map_id)

    if layer is None:
        return mask.astype(bool), {
            "enabled": False,
            "mapId": map_id or "none",
            "name": "No mask",
            "invalidZoneCount": 0,
            "endLocationCount": 0,
        }

    scale_x = width / float(layer.image_size)
    scale_y = height / float(layer.image_size)

    for zone in layer.invalid_zones:
        x = int(round(float(zone["x"]) * scale_x))
        y = int(round(float(zone["y"]) * scale_y))
        radius = int(round(float(zone["radius"]) * (scale_x + scale_y) * 0.5))
        cv2.circle(mask, (x, y), max(1, radius), 0, -1, cv2.LINE_AA)

    return mask.astype(bool), layer.summary(enabled=True)
