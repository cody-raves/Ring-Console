from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path


MAPS = {
    "mp_rr_olympus_mu3": {
        "name": "Olympus",
        "url": "https://apexlegendsstatus.com/interactive-map/mp_rr_olympus_mu3",
    },
    "mp_rr_canyonlands_hu": {
        "name": "Kings Canyon",
        "url": "https://apexlegendsstatus.com/interactive-map/mp_rr_canyonlands_hu",
    },
    "mp_rr_tropic_island_mu2": {
        "name": "Storm Point",
        "url": "https://apexlegendsstatus.com/interactive-map/mp_rr_tropic_island_mu2",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "ring_console" / "data" / "invalid_zones"


def fetch_html(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "RingConsole/0.1 (+local research)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def extract_map_data(html: str) -> dict:
    marker = "let mapData = "
    marker_index = html.find(marker)
    if marker_index < 0:
        raise ValueError("Could not find mapData in the page.")

    start = html.find("{", marker_index)
    if start < 0:
        raise ValueError("Could not find the mapData JSON object.")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(html)):
        char = html[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(html[start : index + 1])

    raise ValueError("mapData JSON object did not terminate.")


def extract_image_size(html: str) -> int:
    south_west = re.search(r"unproject\(\[0,\s*(\d+)\]", html)
    north_east = re.search(r"unproject\(\[(\d+),\s*0\]", html)
    if not south_west or not north_east:
        return 16384

    height = int(south_west.group(1))
    width = int(north_east.group(1))
    if width != height:
        raise ValueError(f"Expected a square map image, got {width}x{height}.")
    return width


def build_payload(map_id: str, meta: dict[str, str], html: str) -> dict:
    map_data = extract_map_data(html)
    return {
        "mapId": map_id,
        "name": meta["name"],
        "sourceUrl": meta["url"],
        "imageSize": extract_image_size(html),
        "invalidEndZones": map_data.get("invalidEndZones", []),
        "endLocation": map_data.get("endLocation", []),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for map_id, meta in MAPS.items():
        html = fetch_html(meta["url"])
        payload = build_payload(map_id, meta, html)
        output_path = DATA_DIR / f"{map_id}.json"
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"{map_id}: {len(payload['invalidEndZones'])} invalid zones, "
            f"{len(payload['endLocation'])} end locations -> {output_path}"
        )


if __name__ == "__main__":
    main()
