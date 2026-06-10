# Ring Console

Local Apex Legends ring prediction prototype.

## Run

```powershell
python -m venv .venv --system-site-packages
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

Open `http://127.0.0.1:5050`, then paste, drop, or upload a full-map screenshot.

## Current Flow

1. Detects and crops the centered map square from a full 16:9 screenshot.
2. Detects the two visible ring circles with OpenCV on the cropped map.
3. Assigns the larger circle as the earlier ring.
4. Predicts rings forward with vector candidates.
5. Draws detected and predicted rings over the cropped map.

Playable terrain masks are stubbed as fully playable for now. The next step is loading a per-map mask and using it in `playable_fraction`.
