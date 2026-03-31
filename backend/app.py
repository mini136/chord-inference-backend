from __future__ import annotations

import io
import json
from collections import Counter, deque
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT_DIR / "model"


class PredictFeaturesRequest(BaseModel):
    features: list[float] = Field(..., min_length=1)
    session_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)
    buffer_size: int = Field(default=7, ge=1, le=51)


class Candidate(BaseModel):
    chord: str
    probability: float


class PredictResponse(BaseModel):
    predictedChord: str
    confidence: float
    topCandidates: list[Candidate]
    stableChord: str


class ModelService:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model: Any | None = None
        self.config: dict[str, Any] = {}
        self.feature_dim = 29
        self.window_ms = 200
        self.hop_ms = 100
        self.start_offset_ms = 300
        self._session_buffers: dict[str, deque[str]] = {}
        self._lock = Lock()

    def load(self) -> None:
        model_path = self.model_dir / "chord_model.joblib"
        config_path = self.model_dir / "model_config.json"
        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"Missing model artifacts in {self.model_dir}")

        self.model = joblib.load(model_path)
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.feature_dim = int(self.config.get("feature_dim", 29))
        self.window_ms = int(self.config.get("window_ms", 200))
        self.hop_ms = int(self.config.get("hop_ms", 100))
        self.start_offset_ms = int(self.config.get("start_offset_ms", 300))

    @property
    def classes(self) -> list[str]:
        if self.model is None:
            return []
        return [str(c) for c in self.model.classes_]

    def _stable(self, session_id: str | None, pred: str, buffer_size: int) -> str:
        if not session_id:
            return pred
        with self._lock:
            buf = self._session_buffers.get(session_id)
            if buf is None or buf.maxlen != buffer_size:
                buf = deque(maxlen=buffer_size)
                self._session_buffers[session_id] = buf
            buf.append(pred)
            return Counter(buf).most_common(1)[0][0]

    def predict_features(self, features: list[float], session_id: str | None, top_k: int, buffer_size: int) -> PredictResponse:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if len(features) != self.feature_dim:
            raise ValueError(f"Expected feature length {self.feature_dim}, got {len(features)}")

        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        probs = self.model.predict_proba(x)[0]

        best_idx = int(np.argmax(probs))
        pred = str(self.model.classes_[best_idx])
        conf = float(probs[best_idx])

        top_idx = np.argsort(probs)[::-1][: min(top_k, len(probs))]
        top = [Candidate(chord=str(self.model.classes_[i]), probability=float(probs[i])) for i in top_idx]

        stable = self._stable(session_id, pred, buffer_size)
        return PredictResponse(predictedChord=pred, confidence=conf, topCandidates=top, stableChord=stable)

    def extract_features_from_window(self, y: np.ndarray, sr: int) -> np.ndarray:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12).mean(axis=1)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        rms = np.array([librosa.feature.rms(y=y).mean()])
        centroid = np.array([librosa.feature.spectral_centroid(y=y, sr=sr).mean()])
        zcr = np.array([librosa.feature.zero_crossing_rate(y).mean()])
        rolloff = np.array([librosa.feature.spectral_rolloff(y=y, sr=sr).mean()])
        return np.concatenate([chroma, mfcc, rms, centroid, zcr, rolloff]).astype(np.float32)

    def split_windows(self, y: np.ndarray, sr: int) -> list[np.ndarray]:
        window = int(sr * self.window_ms / 1000)
        hop = int(sr * self.hop_ms / 1000)
        offset = int(sr * self.start_offset_ms / 1000)
        if len(y) < window:
            return []

        windows: list[np.ndarray] = []
        for start in range(offset, len(y) - window + 1, hop):
            segment = y[start : start + window]
            segment = segment - float(np.mean(segment))
            peak = np.max(np.abs(segment)) + 1e-8
            windows.append(segment / peak)
        return windows

    def predict_wav_bytes(self, wav_bytes: bytes, session_id: str | None, top_k: int, buffer_size: int) -> PredictResponse:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        audio, sr = sf.read(io.BytesIO(wav_bytes), always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        y = np.asarray(audio, dtype=np.float32)
        if y.size == 0:
            raise ValueError("Empty audio payload")

        windows = self.split_windows(y, int(sr))
        if not windows:
            raise ValueError("Audio too short for configured window size")

        feats = np.vstack([self.extract_features_from_window(w, int(sr)) for w in windows])
        probs_all = self.model.predict_proba(feats)
        mean_probs = np.mean(probs_all, axis=0)

        best_idx = int(np.argmax(mean_probs))
        pred = str(self.model.classes_[best_idx])
        conf = float(mean_probs[best_idx])

        top_idx = np.argsort(mean_probs)[::-1][: min(top_k, len(mean_probs))]
        top = [Candidate(chord=str(self.model.classes_[i]), probability=float(mean_probs[i])) for i in top_idx]

        stable = self._stable(session_id, pred, buffer_size)
        return PredictResponse(predictedChord=pred, confidence=conf, topCandidates=top, stableChord=stable)


service = ModelService(DEFAULT_MODEL_DIR)
app = FastAPI(title="Chord Inference Backend", version="1.0.0")


@app.on_event("startup")
def startup_event() -> None:
    service.load()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "classes": service.classes,
        "feature_dim": service.feature_dim,
        "window_ms": service.window_ms,
        "hop_ms": service.hop_ms,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "model_dir": str(service.model_dir),
        "classes": service.classes,
        "config": service.config,
    }


@app.post("/predict-features", response_model=PredictResponse)
def predict_features(req: PredictFeaturesRequest) -> PredictResponse:
    try:
        return service.predict_features(req.features, req.session_id, req.top_k, req.buffer_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict-wav", response_model=PredictResponse)
async def predict_wav(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    top_k: int = Form(default=3),
    buffer_size: int = Form(default=7),
) -> PredictResponse:
    try:
        wav_bytes = await file.read()
        return service.predict_wav_bytes(wav_bytes, session_id, top_k, buffer_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
