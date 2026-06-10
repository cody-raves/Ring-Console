from __future__ import annotations

from flask import Flask, jsonify, request, send_from_directory

from ring_console.vision import analyze_image_bytes


app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 24 * 1024 * 1024


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/analyze")
def analyze():
    upload = request.files.get("image")
    if upload is None:
        return jsonify({"ok": False, "error": "No image was uploaded."}), 400

    first_ring = int(request.form.get("firstRing", "1"))
    stage = request.form.get("stage", "predict")
    map_id = request.form.get("mapId", "mp_rr_olympus_mu3")
    show_invalid_mask = request.form.get("showInvalidMask", "false") == "true"

    try:
        result = analyze_image_bytes(
            upload.read(),
            first_ring=first_ring,
            stage=stage,
            map_id=map_id,
            show_invalid_mask=show_invalid_mask,
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True, use_reloader=False)
