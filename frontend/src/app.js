// Simple frontend logic for live chord prediction
// Assumes backend API endpoints are available at /api/*

const micSelect = document.getElementById("mic");
const startBtn = document.getElementById("startBtn");
const predictionDiv = document
  .getElementById("prediction")
  .querySelector("span");
const apiEndpointsList = document.getElementById("apiEndpoints");

let audioStream = null;
let audioContext = null;

// Fetch available microphones
document.addEventListener("DOMContentLoaded", async () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    micSelect.innerHTML = "<option>Microphone API not supported</option>";
    startBtn.disabled = true;
    return;
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  const mics = devices.filter((d) => d.kind === "audioinput");
  micSelect.innerHTML = "";
  mics.forEach((mic) => {
    const opt = document.createElement("option");
    opt.value = mic.deviceId;
    opt.textContent = mic.label || `Microphone ${mic.deviceId}`;
    micSelect.appendChild(opt);
  });
  if (mics.length === 0) {
    micSelect.innerHTML = "<option>No microphones found</option>";
    startBtn.disabled = true;
  }

  // Fetch and display backend API endpoints
  try {
    const res = await fetch("/api");
    if (res.ok) {
      const endpoints = await res.json();
      apiEndpointsList.innerHTML = "";
      endpoints.forEach((ep) => {
        const li = document.createElement("li");
        li.textContent = ep;
        apiEndpointsList.appendChild(li);
      });
    } else {
      apiEndpointsList.innerHTML = "<li>Could not fetch endpoints</li>";
    }
  } catch {
    apiEndpointsList.innerHTML = "<li>Could not fetch endpoints</li>";
  }
});

startBtn.addEventListener("click", async () => {
  if (startBtn.textContent === "Start Live Prediction") {
    await startLivePrediction();
  } else {
    stopLivePrediction();
  }
});

async function startLivePrediction() {
  startBtn.textContent = "Stop Live Prediction";
  predictionDiv.textContent = "-";
  const deviceId = micSelect.value;
  try {
    audioStream = await navigator.mediaDevices.getUserMedia({
      audio: { deviceId },
    });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(audioStream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    source.connect(processor);
    processor.connect(audioContext.destination);
    processor.onaudioprocess = async (e) => {
      const input = e.inputBuffer.getChannelData(0);
      // Convert Float32Array to Int16Array PCM
      const pcm = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        pcm[i] = Math.max(-32768, Math.min(32767, input[i] * 32767));
      }
      // Send to backend for prediction
      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: pcm.buffer,
        });
        if (res.ok) {
          const { chord } = await res.json();
          predictionDiv.textContent = chord || "-";
        } else {
          predictionDiv.textContent = "Error";
        }
      } catch {
        predictionDiv.textContent = "Error";
      }
    };
    // Save processor to stop later
    window._audioProcessor = processor;
  } catch (err) {
    predictionDiv.textContent = "Mic error";
    startBtn.textContent = "Start Live Prediction";
  }
}

function stopLivePrediction() {
  startBtn.textContent = "Start Live Prediction";
  predictionDiv.textContent = "-";
  if (window._audioProcessor) {
    window._audioProcessor.disconnect();
    window._audioProcessor = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (audioStream) {
    audioStream.getTracks().forEach((track) => track.stop());
    audioStream = null;
  }
}
