// Live Chord Prediction – frontend logic
// Backend runs on localhost:40150 (FastAPI)

const BACKEND = "http://localhost:40150";
const BUFFER_SECONDS = 1.0; // how much audio to accumulate before sending
const SEND_INTERVAL_MS = 800; // how often to send predictions

const micSelect = document.getElementById("mic");
const startBtn = document.getElementById("startBtn");
const predictionSpan = document.getElementById("prediction");
const stableChordSpan = document.getElementById("stableChord");
const confidenceSpan = document.getElementById("confidence");
const topCandidatesDiv = document.getElementById("topCandidates");
const modelInfoList = document.getElementById("modelInfo");
const statusBar = document.getElementById("status-bar");

let audioStream = null;
let audioContext = null;
let sendTimer = null;
let audioBuffer = [];
let sessionId = "session_" + Date.now();
let backendReady = false;

// ---- helpers ----

function setStatus(ok) {
  backendReady = ok;
  statusBar.textContent = ok ? "Backend: připojeno" : "Backend: odpojeno";
  statusBar.className =
    "status-bar " + (ok ? "status-connected" : "status-disconnected");
  startBtn.disabled = !ok;
}

function encodeWav(samples, sampleRate) {
  const numSamples = samples.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);
  function writeStr(offset, str) {
    for (let i = 0; i < str.length; i++)
      view.setUint8(offset + i, str.charCodeAt(i));
  }
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, "data");
  view.setUint32(40, numSamples * 2, true);
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 32768 : s * 32767, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

// ---- init ----

window.addEventListener("DOMContentLoaded", async () => {
  // check backend health
  try {
    const res = await fetch(BACKEND + "/health");
    if (res.ok) {
      const data = await res.json();
      setStatus(true);
      modelInfoList.innerHTML = "";
      const items = [
        "Akordy: " + (data.classes || []).join(", "),
        "Feature dim: " + data.feature_dim,
        "Window: " + data.window_ms + " ms",
        "Hop: " + data.hop_ms + " ms",
      ];
      items.forEach((txt) => {
        const li = document.createElement("li");
        li.textContent = txt;
        modelInfoList.appendChild(li);
      });
    } else {
      setStatus(false);
    }
  } catch {
    setStatus(false);
  }

  // enumerate microphones
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    micSelect.innerHTML = "<option>Mikrofon API není podporováno</option>";
    startBtn.disabled = true;
    return;
  }
  // need permission first to get labels
  try {
    const tempStream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });
    tempStream.getTracks().forEach((t) => t.stop());
  } catch {
    // user denied – still try enumeration
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  const mics = devices.filter((d) => d.kind === "audioinput");
  micSelect.innerHTML = "";
  mics.forEach((mic, idx) => {
    const opt = document.createElement("option");
    opt.value = mic.deviceId;
    opt.textContent = mic.label || "Mikrofon " + (idx + 1);
    micSelect.appendChild(opt);
  });
  if (mics.length === 0) {
    micSelect.innerHTML = "<option>Žádný mikrofon nenalezen</option>";
    startBtn.disabled = true;
  }
});

// ---- start/stop ----

startBtn.addEventListener("click", async () => {
  if (startBtn.textContent === "Start Live Prediction") {
    await startLivePrediction();
  } else {
    stopLivePrediction();
  }
});

async function startLivePrediction() {
  startBtn.textContent = "Stop Live Prediction";
  predictionSpan.textContent = "-";
  stableChordSpan.textContent = "-";
  confidenceSpan.textContent = "-";
  topCandidatesDiv.innerHTML = "";
  audioBuffer = [];
  sessionId = "session_" + Date.now();

  const deviceId = micSelect.value;
  try {
    audioStream = await navigator.mediaDevices.getUserMedia({
      audio: { deviceId: deviceId ? { exact: deviceId } : undefined },
    });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(audioStream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    source.connect(processor);
    processor.connect(audioContext.destination);

    // accumulate audio samples
    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      const copy = new Float32Array(input.length);
      copy.set(input);
      audioBuffer.push(copy);
    };
    window._audioProcessor = processor;

    // periodically send accumulated audio to backend
    sendTimer = setInterval(() => sendAudioToBackend(), SEND_INTERVAL_MS);
  } catch (err) {
    predictionSpan.textContent = "Chyba mikrofonu";
    startBtn.textContent = "Start Live Prediction";
  }
}

async function sendAudioToBackend() {
  if (audioBuffer.length === 0 || !audioContext) return;

  // collect buffered samples
  const chunks = audioBuffer.splice(0, audioBuffer.length);
  const totalLen = chunks.reduce((acc, c) => acc + c.length, 0);
  const merged = new Float32Array(totalLen);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  // need at least ~0.5s of audio
  const minSamples = audioContext.sampleRate * 0.5;
  if (merged.length < minSamples) return;

  const wavBlob = encodeWav(merged, audioContext.sampleRate);
  const formData = new FormData();
  formData.append("file", wavBlob, "audio.wav");
  formData.append("session_id", sessionId);
  formData.append("top_k", "5");
  formData.append("buffer_size", "7");

  try {
    const res = await fetch(BACKEND + "/predict-wav", {
      method: "POST",
      body: formData,
    });
    if (res.ok) {
      const data = await res.json();
      predictionSpan.textContent = data.predictedChord || "-";
      stableChordSpan.textContent = data.stableChord || "-";
      confidenceSpan.textContent = (data.confidence * 100).toFixed(1) + " %";

      topCandidatesDiv.innerHTML = "";
      if (data.topCandidates && data.topCandidates.length > 0) {
        data.topCandidates.forEach((c) => {
          const tag = document.createElement("span");
          tag.className = "candidate-tag";
          tag.textContent =
            c.chord + " " + (c.probability * 100).toFixed(0) + " %";
          topCandidatesDiv.appendChild(tag);
        });
      }
    } else {
      predictionSpan.textContent = "Chyba serveru";
    }
  } catch {
    predictionSpan.textContent = "Chyba spojení";
  }
}

function stopLivePrediction() {
  startBtn.textContent = "Start Live Prediction";
  if (sendTimer) {
    clearInterval(sendTimer);
    sendTimer = null;
  }
  audioBuffer = [];
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
