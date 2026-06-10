const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");
const cropButton = document.querySelector("#cropButton");
const detectButton = document.querySelector("#detectButton");
const predictButton = document.querySelector("#predictButton");
const mapId = document.querySelector("#mapId");
const showInvalidMask = document.querySelector("#showInvalidMask");
const firstRing = document.querySelector("#firstRing");
const fileName = document.querySelector("#fileName");
const statusBox = document.querySelector("#statusBox");
const previewImage = document.querySelector("#previewImage");
const resultImage = document.querySelector("#resultImage");
const emptyState = document.querySelector("#emptyState");
const cropInfo = document.querySelector("#cropInfo");
const ringList = document.querySelector("#ringList");
const predictionList = document.querySelector("#predictionList");
const candidateList = document.querySelector("#candidateList");

let activeFile = null;
let isRunning = false;
let lastCompletedStage = "input";

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) {
    setImageFile(file);
    runStage("crop");
  }
});

document.addEventListener("paste", (event) => {
  const items = Array.from(event.clipboardData?.items || []);
  const imageItem = items.find((item) => item.type.startsWith("image/"));
  if (!imageItem) {
    return;
  }

  const file = imageItem.getAsFile();
  if (file) {
    setImageFile(file, "clipboard-screenshot.png");
    runStage("crop");
  }
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("is-dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("is-dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");

  const file = Array.from(event.dataTransfer.files).find((item) => item.type.startsWith("image/"));
  if (file) {
    setImageFile(file);
    runStage("crop");
  }
});

cropButton.addEventListener("click", () => runStage("crop"));
detectButton.addEventListener("click", () => runStage("detect"));
predictButton.addEventListener("click", () => runStage("predict"));

firstRing.addEventListener("change", () => {
  lastCompletedStage = activeFile ? "crop" : "input";
  predictionList.innerHTML = "";
  updateButtons();
});

mapId.addEventListener("change", () => {
  rerunCurrentStage();
});

showInvalidMask.addEventListener("change", () => {
  rerunCurrentStage();
});

function setImageFile(file, forcedName = null) {
  activeFile = file;
  lastCompletedStage = "input";
  fileName.textContent = forcedName || file.name || "clipboard-screenshot.png";
  statusBox.textContent = "Image loaded.";
  cropInfo.innerHTML = "";
  ringList.innerHTML = "";
  predictionList.innerHTML = "";
  candidateList.innerHTML = "";
  resultImage.removeAttribute("src");
  resultImage.classList.remove("is-visible");
  emptyState.hidden = true;

  const previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewImage.classList.add("is-visible");
  updateButtons();
}

async function runStage(stage) {
  if (!activeFile || isRunning) {
    return;
  }

  isRunning = true;
  updateButtons(stage);
  statusBox.textContent = stageStatus(stage);

  if (stage === "crop") {
    cropInfo.innerHTML = "";
    ringList.innerHTML = "";
    predictionList.innerHTML = "";
    candidateList.innerHTML = "";
  } else if (stage === "detect") {
    ringList.innerHTML = "";
    predictionList.innerHTML = "";
    candidateList.innerHTML = "";
  } else if (stage === "predict") {
    predictionList.innerHTML = "";
    candidateList.innerHTML = "";
  }

  const form = new FormData();
  form.append("image", activeFile, activeFile.name || "screenshot.png");
  form.append("firstRing", firstRing.value);
  form.append("mapId", mapId.value);
  form.append("showInvalidMask", showInvalidMask.checked ? "true" : "false");
  form.append("stage", stage);

  const controller = new AbortController();
  const timeoutMs = stage === "crop" ? 5000 : stage === "predict" ? 25000 : 15000;
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: form,
      signal: controller.signal,
    });

    const result = await response.json();
    showStageResult(stage, result, response.ok && result.ok);
  } catch (error) {
    statusBox.textContent =
      error.name === "AbortError"
        ? `${stageLabel(stage)} timed out.`
        : `${stageLabel(stage)} failed: ${error.message}`;
  } finally {
    window.clearTimeout(timeoutId);
    isRunning = false;
    updateButtons();
  }
}

function showStageResult(stage, result, isSuccess) {
  if (result.overlay) {
    resultImage.src = result.overlay;
    resultImage.classList.add("is-visible");
    previewImage.classList.remove("is-visible");
  }

  if (result.crop) {
    renderCropInfo(result.crop, result.mask);
  }

  renderMetricList(ringList, result.rings || []);
  renderMetricList(predictionList, result.predictions || []);
  renderCandidateList(candidateList, result.candidates || []);

  if (!isSuccess) {
    statusBox.textContent = result.error || `${stageLabel(stage)} failed.`;
    lastCompletedStage = stage === "crop" ? "input" : "crop";
    return;
  }

  lastCompletedStage = stage;

  if (stage === "crop") {
    statusBox.textContent = `Crop ready: ${result.crop.width} x ${result.crop.height}.`;
  } else if (stage === "detect") {
    statusBox.textContent = `Detected ${result.rings.length} rings in ${result.elapsedMs} ms.`;
  } else {
    const predictions = result.predictions || [];
    const firstPrediction = predictions[0];
    const lastPrediction = predictions[predictions.length - 1];
    statusBox.textContent = firstPrediction
      ? `Predicted R${firstPrediction.number}-R${lastPrediction.number} from detected rings in ${result.elapsedMs} ms.`
      : `No next ring to predict.`;
  }
}

function renderCropInfo(crop, mask) {
  cropInfo.innerHTML = "";
  const row = document.createElement("div");
  row.className = "metric-row";
  row.innerHTML = `
    <strong>${crop.mode === "detected-map" ? "Map" : "Fallback"}</strong>
    <span>${crop.width} x ${crop.height}</span>
    <span>${crop.x}, ${crop.y}</span>
  `;
  cropInfo.appendChild(row);

  if (mask) {
    const maskRow = document.createElement("div");
    maskRow.className = "metric-row";
    maskRow.innerHTML = `
      <strong>Mask</strong>
      <span>${mask.name || "No mask"}</span>
      <span>${mask.enabled ? `${mask.invalidZoneCount} invalid` : "off"}</span>
    `;
    cropInfo.appendChild(maskRow);
  }
}

function rerunCurrentStage() {
  if (!activeFile || isRunning) {
    return;
  }

  const stage = lastCompletedStage === "input" ? "crop" : lastCompletedStage;
  runStage(stage);
}

function renderMetricList(container, rings) {
  container.innerHTML = "";

  if (!rings || rings.length === 0) {
    container.innerHTML = "<div class=\"metric-row muted\">None</div>";
    return;
  }

  for (const ring of rings) {
    const row = document.createElement("div");
    row.className = "metric-row";
    const source = ring.source?.startsWith("prediction:")
      ? ring.source.replace("prediction:", "").replaceAll("-", " ")
      : "";
    const detail = typeof ring.gap === "number"
      ? `r ${Math.round(ring.radius)} g ${Math.round(ring.gap)}`
      : `r ${Math.round(ring.radius)}`;
    row.innerHTML = `
      <strong>R${ring.number}</strong>
      <span>${Math.round(ring.cropX)}, ${Math.round(ring.cropY)}</span>
      <span>${source ? `${detail} ${source}` : detail}</span>
    `;
    container.appendChild(row);
  }
}

function renderCandidateList(container, candidates) {
  container.innerHTML = "";

  const visible = (candidates || []).filter((candidate) => !candidate.source.startsWith("prediction"));
  if (visible.length === 0) {
    container.innerHTML = "<div class=\"metric-row muted\">None</div>";
    return;
  }

  for (const candidate of visible) {
    const row = document.createElement("div");
    row.className = "metric-row";
    const source = candidate.source.replaceAll("-", " ");
    const detail = typeof candidate.gap === "number"
      ? `g ${Math.round(candidate.gap)}`
      : `s ${candidate.score.toFixed(2)}`;
    row.innerHTML = `
      <strong>${source}</strong>
      <span>R${candidate.number} ${Math.round(candidate.cropX)}, ${Math.round(candidate.cropY)}</span>
      <span>${detail}</span>
    `;
    container.appendChild(row);
  }
}

function updateButtons(activeStage = null) {
  cropButton.disabled = !activeFile || isRunning;
  detectButton.disabled = !activeFile || isRunning || lastCompletedStage === "input";
  predictButton.disabled =
    !activeFile || isRunning || !["detect", "predict"].includes(lastCompletedStage);

  for (const [button, stage] of [
    [cropButton, "crop"],
    [detectButton, "detect"],
    [predictButton, "predict"],
  ]) {
    button.classList.toggle("is-active", activeStage === stage);
  }
}

function stageStatus(stage) {
  if (stage === "crop") {
    return "Detecting map crop...";
  }
  if (stage === "detect") {
    return "Detecting rings...";
  }
  return "Predicting rings...";
}

function stageLabel(stage) {
  if (stage === "crop") {
    return "Crop";
  }
  if (stage === "detect") {
    return "Detection";
  }
  return "Prediction";
}
