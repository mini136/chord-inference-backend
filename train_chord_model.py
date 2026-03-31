import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


WINDOW_MS = 200
HOP_MS = 100
START_OFFSET_MS = 300


def extract_features_from_window(y: np.ndarray, sr: int) -> np.ndarray:
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12).mean(axis=1)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    rms = np.array([librosa.feature.rms(y=y).mean()])
    centroid = np.array([librosa.feature.spectral_centroid(y=y, sr=sr).mean()])
    zcr = np.array([librosa.feature.zero_crossing_rate(y).mean()])
    rolloff = np.array([librosa.feature.spectral_rolloff(y=y, sr=sr).mean()])
    return np.concatenate([chroma, mfcc, rms, centroid, zcr, rolloff]).astype(np.float32)


def split_windows(
    y: np.ndarray,
    sr: int,
    window_ms: int = WINDOW_MS,
    hop_ms: int = HOP_MS,
    start_offset_ms: int = START_OFFSET_MS,
) -> List[np.ndarray]:
    window = int(sr * window_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    offset = int(sr * start_offset_ms / 1000)
    if len(y) < window:
        return []

    windows = []
    for start in range(offset, len(y) - window + 1, hop):
        segment = y[start:start + window]
        segment = segment - float(np.mean(segment))
        peak = np.max(np.abs(segment)) + 1e-8
        segment = segment / peak
        windows.append(segment)
    return windows


def build_dataset_from_rows(rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if rows.empty:
        raise RuntimeError("No rows provided to build dataset.")

    X: List[np.ndarray] = []
    y: List[str] = []

    for _, row in rows.iterrows():
        wav_path = Path(str(row["wav_path"]))
        label = str(row["label"])
        signal, sr_loaded = librosa.load(wav_path, sr=None, mono=True)
        sr = int(sr_loaded)

        for window in split_windows(signal, sr):
            X.append(extract_features_from_window(window, sr))
            y.append(label)

    if not X:
        raise RuntimeError("No training windows generated. Check recordings.")

    return np.vstack(X), np.array(y)


def collect_rows_from_single_session(session_dir: Path) -> pd.DataFrame:
    metadata_path = session_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    df = pd.read_csv(metadata_path)[["relative_path", "label"]].copy()
    df["session"] = session_dir.name
    df["wav_path"] = df["relative_path"].map(lambda rel: str((session_dir / rel).resolve()))
    return df[["session", "wav_path", "label"]]


def collect_rows_from_dataset_dir(dataset_dir: Path) -> pd.DataFrame:
    sessions = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    all_rows: List[pd.DataFrame] = []

    for session in sessions:
        metadata_path = session / "metadata.csv"
        if not metadata_path.exists():
            continue
        session_df = pd.read_csv(metadata_path)[["relative_path", "label"]].copy()
        session_df["session"] = session.name
        session_df["wav_path"] = session_df["relative_path"].map(lambda rel: str((session / rel).resolve()))
        all_rows.append(session_df[["session", "wav_path", "label"]])

    if not all_rows:
        raise RuntimeError(f"No sessions with metadata.csv found in: {dataset_dir}")

    return pd.concat(all_rows, ignore_index=True)


def generate_binary_plots(model: Pipeline, X_all: np.ndarray, y_all: np.ndarray, out_dir: Path) -> bool:
    clf = model.named_steps["clf"]
    model_classes = [str(c) for c in clf.classes_]

    if len(model_classes) != 2:
        print("Skipping logistic plots: require exactly 2 classes.")
        return False

    positive_class = model_classes[1]
    negative_class = model_classes[0]

    z = model.decision_function(X_all)
    z = np.asarray(z).reshape(-1)
    p = 1.0 / (1.0 + np.exp(-z))

    y_num = np.array([1 if str(lbl) == positive_class else 0 for lbl in y_all], dtype=int)
    idx = np.arange(len(z))

    colors = np.where(y_num == 1, "#d62728", "#1f77b4")
    labels = np.where(y_num == 1, positive_class, negative_class)

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    for class_name in [negative_class, positive_class]:
        mask = labels == class_name
        ax1.scatter(idx[mask], z[mask], s=10, alpha=0.7, label=class_name)
    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax1.set_title("Lineární skóre logistické regrese (všechny vzorky)")
    ax1.set_xlabel("Index vzorku")
    ax1.set_ylabel("Lineární skóre z")
    ax1.legend(title="Label")
    ax1.grid(alpha=0.2)
    fig1.tight_layout()
    fig1.savefig(out_dir / "plot_linear_score.png", dpi=150)
    plt.close(fig1)

    x_curve = np.linspace(float(np.min(z)) - 1.0, float(np.max(z)) + 1.0, 400)
    y_curve = 1.0 / (1.0 + np.exp(-x_curve))

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(x_curve, y_curve, color="black", linewidth=2, label="sigmoid(z)")
    for class_name in [negative_class, positive_class]:
        mask = labels == class_name
        ax2.scatter(z[mask], p[mask], s=10, alpha=0.7, label=f"vzorky {class_name}")
    ax2.set_title("Sigmoid převod lineárního skóre na pravděpodobnost (všechny vzorky)")
    ax2.set_xlabel("Lineární skóre z")
    ax2.set_ylabel(f"P({positive_class})")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend()
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(out_dir / "plot_sigmoid_with_samples.png", dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    bins = 40
    ax3.hist(z[y_num == 0], bins=bins, alpha=0.6, label=negative_class, color="#1f77b4", density=True)
    ax3.hist(z[y_num == 1], bins=bins, alpha=0.6, label=positive_class, color="#d62728", density=True)
    ax3.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax3.set_title("Histogram lineárního skóre podle tříd")
    ax3.set_xlabel("Lineární skóre z")
    ax3.set_ylabel("Hustota")
    ax3.legend(title="Label")
    ax3.grid(alpha=0.2)
    fig3.tight_layout()
    fig3.savefig(out_dir / "plot_linear_score_histogram.png", dpi=150)
    plt.close(fig3)
    return True


def generate_multiclass_plots(
    y_test: np.ndarray,
    preds: np.ndarray,
    y_all: np.ndarray,
    classes: list[str],
    out_dir: Path,
) -> None:
    cm = confusion_matrix(y_test, preds, labels=classes)
    cm_norm = confusion_matrix(y_test, preds, labels=classes, normalize="true")

    fig_cm, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title("Confusion Matrix (počty)")
    axes[0].set_xlabel("Predikce")
    axes[0].set_ylabel("Skutečnost")
    axes[0].set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    axes[0].set_yticks(range(len(classes)), classes)
    fig_cm.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=9)

    im1 = axes[1].imshow(cm_norm, interpolation="nearest", cmap="Greens", vmin=0.0, vmax=1.0)
    axes[1].set_title("Confusion Matrix (normalizovaná)")
    axes[1].set_xlabel("Predikce")
    axes[1].set_ylabel("Skutečnost")
    axes[1].set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    axes[1].set_yticks(range(len(classes)), classes)
    fig_cm.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            axes[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)

    fig_cm.tight_layout()
    fig_cm.savefig(out_dir / "plot_confusion_matrix.png", dpi=150)
    plt.close(fig_cm)

    values, counts = np.unique(y_all, return_counts=True)
    count_map = {str(v): int(c) for v, c in zip(values, counts)}
    ordered_counts = [count_map.get(cls, 0) for cls in classes]

    fig_dist, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(classes, ordered_counts, color="#4c78a8")
    ax.set_title("Rozložení tříd v datasetu (po oknech)")
    ax.set_xlabel("Třída")
    ax.set_ylabel("Počet oken")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, ordered_counts):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, str(value), ha="center", va="bottom", fontsize=9)
    fig_dist.tight_layout()
    fig_dist.savefig(out_dir / "plot_class_distribution.png", dpi=150)
    plt.close(fig_dist)


def get_classifier(model_type: str):
    if model_type == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    elif model_type == "forest":
        return RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train simple chord classifier (A/Am etc.)")
    parser.add_argument("--session-dir", help="Path to one dataset session dir")
    parser.add_argument("--dataset-dir", help="Path to dataset dir with multiple sessions")
    parser.add_argument("--out-dir", default="model", help="Output dir for model artifacts")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split ratio")
    parser.add_argument("--model-type", choices=["logreg", "forest"], default="logreg", help="Type of model to train")
    args = parser.parse_args()

    if bool(args.session_dir) == bool(args.dataset_dir):
        raise RuntimeError("Use exactly one of --session-dir or --dataset-dir")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).resolve()
        files_df = collect_rows_from_dataset_dir(dataset_dir)
        source_desc = str(dataset_dir)
    else:
        session_dir = Path(args.session_dir).resolve()
        files_df = collect_rows_from_single_session(session_dir)
        source_desc = str(session_dir)

    classes = sorted(files_df["label"].astype(str).unique().tolist())

    if files_df["label"].nunique() < 2:
        raise RuntimeError("Need at least 2 classes in metadata for training.")

    can_stratify_files = all(count >= 2 for count in files_df["label"].value_counts().tolist())
    train_rows, test_rows = train_test_split(
        files_df,
        test_size=args.test_size,
        random_state=42,
        stratify=files_df["label"] if can_stratify_files else None,
    )

    X_train, y_train = build_dataset_from_rows(train_rows)
    X_test, y_test = build_dataset_from_rows(test_rows)
    X_all, y_all = build_dataset_from_rows(files_df)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", get_classifier(args.model_type)),
    ])

    model.fit(X_train, y_train)
    preds = np.asarray(model.predict(X_test))

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, preds, zero_division=0)

    joblib.dump(model, out_dir / "chord_model.joblib")
    plots_generated = generate_binary_plots(model, X_all, y_all, out_dir)

    config = {
        "classes": classes,
        "window_ms": WINDOW_MS,
        "hop_ms": HOP_MS,
        "start_offset_ms": START_OFFSET_MS,
        "feature_dim": int(X_all.shape[1]),
        "num_windows": int(X_all.shape[0]),
        "num_windows_train": int(X_train.shape[0]),
        "num_windows_test": int(X_test.shape[0]),
        "source": source_desc,
        "num_sessions": int(files_df["session"].nunique()),
        "num_files": int(len(files_df)),
    }
    (out_dir / "model_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== TRAINING DONE ===")
    print(f"Source: {source_desc}")
    print(f"Sessions: {files_df['session'].nunique()}, Files: {len(files_df)}")
    print(f"Windows train/test/all: {X_train.shape[0]}/{X_test.shape[0]}/{X_all.shape[0]}")
    print(f"Feature dim: {X_all.shape[1]}")
    print(f"Classes: {classes}")
    print(report_text)
    print(f"Saved model: {out_dir / 'chord_model.joblib'}")
    if plots_generated:
        print(f"Saved plot: {out_dir / 'plot_linear_score.png'}")
        print(f"Saved plot: {out_dir / 'plot_sigmoid_with_samples.png'}")
        print(f"Saved plot: {out_dir / 'plot_linear_score_histogram.png'}")
    else:
        generate_multiclass_plots(y_test, preds, y_all, classes, out_dir)
        print("Binary logistic plots were skipped (model has more than 2 classes).")
        print(f"Saved plot: {out_dir / 'plot_confusion_matrix.png'}")
        print(f"Saved plot: {out_dir / 'plot_class_distribution.png'}")


if __name__ == "__main__":
    main()
