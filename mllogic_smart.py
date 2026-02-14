"""
Keno Predictor Logic - Smart Model (Upgraded)
================================================

Advanced ensemble model with:
- Multi-input architecture (sequence + global + draw-specific)
- Attention mechanisms for sequence inputs
- Cross-validation for time-series data
- Advanced evaluation metrics (Precision@K, Recall@K, Hit Rate)
- Dynamic retraining schedules

Author: Keno Prediction System
Version: 2.0
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import csv
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from collections import Counter, defaultdict
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import TimeSeriesSplit
import random

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_FILE_PATH = "games.csv"
MODEL_SAVE_PATH = "modelsmart.keras"
PREDICTIONS_SAVE_FILE = "pred_smart.json"
BRAINSTATE_FILE = "brainstate.json"

# Training configuration
FULL_RETRAIN_INTERVAL = 12
MINI_EPOCHS = 5
FULL_EPOCHS = 130
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

# Feature lookback windows
SHORT_LOOK = 10
MEDIUM_LOOK = 30
LONG_LOOK = 100
MAX_KENO_NUMBER = 80

# Sequence processing
SEQUENCE_LENGTH = 20

# Set building config
CORE_N = 20
SET_SIZES = (3, 5, 8)
CANDIDATES_PER_SIZE = 30
TUNE_SAMPLES = 120
TUNE_TEMPS = (0.7, 1.0, 1.3)
TUNE_MUS = (0.0, 0.03, 0.06, 0.10)
RNG_SEED = 2468

EXTRA_STYLE_SIZES = (10,)

# Cross-validation
CV_FOLDS = 5
CV_MIN_SAMPLES = 500

# ==============================================================================
# CUSTOM LOSS FUNCTIONS
# ==============================================================================

@tf.keras.utils.register_keras_serializable()
def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal Loss for addressing class imbalance."""
    y_true = tf.cast(y_true, tf.float32)
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    bce_pos = -y_true * tf.math.log(y_pred)
    bce_neg = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)

    pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
    focal_weight = alpha_t * tf.pow(1.0 - pt, gamma)

    return tf.reduce_sum(focal_weight * (bce_pos + bce_neg), axis=-1)


@tf.keras.utils.register_keras_serializable()
def precision_recall_loss(y_true, y_pred):
    """Combined precision-recall loss for better top-K performance."""
    y_true = tf.cast(y_true, tf.float32)
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    # Binary cross-entropy
    bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

    # Top-K penalty: encourage high values for true positives
    k = tf.minimum(20, tf.reduce_sum(y_true))
    top_k_values = tf.math.top_k(y_pred, k=k)[0]
    top_k_targets = tf.gather_nd(y_true, tf.math.top_k(y_pred, k=k)[1])

    top_k_loss = tf.reduce_mean(top_k_targets * (1 - top_k_values))

    return tf.reduce_mean(bce) + 0.1 * top_k_loss


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_brain_bias() -> List[float]:
    """Load global bias weights from brainstate file."""
    bias = [1.0] * 81
    if not os.path.exists(BRAINSTATE_FILE):
        return bias
    try:
        with open(BRAINSTATE_FILE, "r") as f:
            state = json.load(f)
        gb = state.get("global_bias", None)
        if isinstance(gb, list) and len(gb) >= 81:
            return [float(x) for x in gb[:81]]
    except Exception:
        pass
    return bias


def load_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load Keno data from canonical CSV format.
    Returns ids, draws, count.
    """
    if not os.path.exists(filepath):
        return np.array([]), np.array([]), 0

    ids, draws = [], []
    try:
        with open(filepath, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                did = r.get("draw_id", "")
                if did is None or str(did).strip() == "":
                    continue
                draw_id = int(str(did).strip())
                nums = [int(r[f"n{i}"]) for i in range(1, 21)]
                if len(nums) != 20:
                    continue
                ids.append(draw_id)
                draws.append(nums)
    except Exception:
        pass

    return np.array(ids, dtype=int), np.array(draws, dtype=int), len(ids)


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

def build_relation_matrix(data: np.ndarray, lookback: int = 250) -> np.ndarray:
    """
    Build co-occurrence matrix for number pairs.
    Used for set optimization and correlation penalties.
    """
    matrix = np.zeros((MAX_KENO_NUMBER + 1, MAX_KENO_NUMBER + 1), dtype=float)
    recent = data[-lookback:] if len(data) >= lookback else data

    for draw in recent:
        for i in draw:
            if i <= 0 or i > MAX_KENO_NUMBER:
                continue
            for j in draw:
                if j <= 0 or j > MAX_KENO_NUMBER or i == j:
                    continue
                matrix[i, j] += 1.0

    mx = float(np.max(matrix))
    if mx > 0:
        matrix /= mx
    return matrix


def compute_advanced_features(history: np.ndarray) -> np.ndarray:
    """
    Compute advanced features for each number including:
    - Frequency in multiple windows
    - Time since last appearance
    - Average gap between appearances
    - Consecutive appearance pattern
    - Positional tendencies
    """
    feats = np.zeros((MAX_KENO_NUMBER, 12), dtype=float)

    if len(history) == 0:
        return feats.flatten()

    # Window-based frequencies
    sh = history[-SHORT_LOOK:].flatten() if len(history) >= SHORT_LOOK else history.flatten()
    mh = history[-MEDIUM_LOOK:].flatten() if len(history) >= MEDIUM_LOOK else history.flatten()
    lh = history[-LONG_LOOK:].flatten() if len(history) >= LONG_LOOK else history.flatten()

    sc = Counter(sh)
    mc = Counter(mh)
    lc = Counter(lh)

    # Time since last appearance and gap analysis
    last_seen = {}
    gaps = defaultdict(list)
    for i, draw in enumerate(history):
        for n in draw:
            if 1 <= n <= MAX_KENO_NUMBER:
                if n in last_seen:
                    gaps[n].append(i - last_seen[n])
                last_seen[n] = i

    # Positional analysis
    position_sum = defaultdict(int)
    position_count = defaultdict(int)
    for draw in history[-MEDIUM_LOOK:]:
        for pos, n in enumerate(draw):
            if 1 <= n <= MAX_KENO_NUMBER:
                position_sum[n] += pos
                position_count[n] += 1

    for n in range(1, MAX_KENO_NUMBER + 1):
        idx = n - 1

        # Frequencies
        feats[idx, 0] = sc.get(n, 0) / max(1, SHORT_LOOK)
        feats[idx, 1] = mc.get(n, 0) / max(1, MEDIUM_LOOK)
        feats[idx, 2] = lc.get(n, 0) / max(1, LONG_LOOK)

        # Time since last appearance (normalized)
        last = last_seen.get(n, -1)
        if last >= 0:
            feats[idx, 3] = (len(history) - 1 - last) / max(1, MEDIUM_LOOK)
        else:
            feats[idx, 3] = 1.0

        # Average gap
        if gaps.get(n):
            feats[idx, 4] = np.mean(gaps[n]) / max(1, MEDIUM_LOOK)
        else:
            feats[idx, 4] = 1.0

        # Gap trend (is gap increasing or decreasing?)
        if len(gaps.get(n, [])) >= 2:
            recent_gaps = gaps[n][-5:]
            feats[idx, 5] = (recent_gaps[-1] - recent_gaps[0]) / max(1, MEDIUM_LOOK)
        else:
            feats[idx, 5] = 0.0

        # Position tendency (normalized 0-1)
        if position_count[n] > 0:
            feats[idx, 6] = position_sum[n] / (position_count[n] * 20.0)
        else:
            feats[idx, 6] = 0.5

        # Zone features
        feats[idx, 7] = 1.0 if n <= 20 else 0.0
        feats[idx, 8] = 1.0 if 20 < n <= 40 else 0.0
        feats[idx, 9] = 1.0 if 40 < n <= 60 else 0.0
        feats[idx, 10] = 1.0 if n > 60 else 0.0

        # Parity
        feats[idx, 11] = 1.0 if n % 2 == 0 else 0.0

    return feats.flatten()


def compute_draw_specific_features(draw: np.ndarray) -> np.ndarray:
    """
    Compute features specific to a single draw.
    Includes statistical properties of the numbers.
    """
    feats = np.zeros(10, dtype=float)

    if draw.size == 0:
        return feats

    valid = draw[(draw >= 1) & (draw <= MAX_KENO_NUMBER)]
    if valid.size == 0:
        return feats

    # Basic stats
    feats[0] = float(np.mean(valid)) / 80.0  # Mean normalized
    feats[1] = float(np.std(valid)) / 80.0  # Std normalized
    feats[2] = float(np.sum(valid <= 40)) / 20.0  # Ratio in first half
    feats[3] = float(np.sum(valid > 40)) / 20.0  # Ratio in second half
    feats[4] = float(np.sum(valid % 2 == 0)) / 20.0  # Ratio even
    feats[5] = float(np.sum(valid % 2 != 0)) / 20.0  # Ratio odd
    feats[6] = float(len(np.unique(valid))) / 20.0  # Uniqueness

    # Range features
    feats[7] = (np.max(valid) - np.min(valid)) / 79.0  # Spread

    # Quadrant distribution
    quadrants = [0, 0, 0, 0]
    for n in valid:
        if n <= 20:
            quadrants[0] += 1
        elif n <= 40:
            quadrants[1] += 1
        elif n <= 60:
            quadrants[2] += 1
        else:
            quadrants[3] += 1
    feats[8] = max(quadrants) / 20.0  # Max quadrant concentration
    feats[9] = len([q for q in quadrants if q >= 3]) / 4.0  # Balanced quadrants

    return feats


def compute_sequence_features(history: np.ndarray, seq_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    Compute sequence features for LSTM/attention processing.
    Returns binary matrix of which numbers appeared in each draw.
    """
    if len(history) < seq_len:
        # Pad with zeros
        padded = np.zeros((seq_len, 20), dtype=int)
        start = seq_len - len(history)
        for i, draw in enumerate(history):
            for j, n in enumerate(draw):
                if 1 <= n <= MAX_KENO_NUMBER:
                    padded[start + i, j] = n
        history = padded

    recent = history[-seq_len:]
    seq_matrix = np.zeros((seq_len, MAX_KENO_NUMBER), dtype=float)

    for i, draw in enumerate(recent):
        for n in draw:
            if 1 <= n <= MAX_KENO_NUMBER:
                seq_matrix[i, n - 1] = 1.0

    return seq_matrix


# ==============================================================================
# DATASET CREATION
# ==============================================================================

def create_dataset(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training dataset with multi-input features.
    Returns X_seq, X_glob, X_spec, Y
    """
    lookback = max(SHORT_LOOK, MEDIUM_LOOK, SEQUENCE_LENGTH)

    if len(data) < lookback + 2:
        return np.array([]), np.array([]), np.array([]), np.array([])

    X_seq, X_glob, X_spec, Y = [], [], [], []

    for i in range(lookback, len(data) - 1):
        # Sequence input
        X_seq.append(compute_sequence_features(data[i - lookback : i]))

        # Global features
        X_glob.append(compute_advanced_features(data[i - lookback : i]))

        # Draw-specific features
        X_spec.append(compute_draw_specific_features(data[i]))

        # Target
        tgt = np.zeros(MAX_KENO_NUMBER, dtype=int)
        for n in data[i + 1]:
            if 1 <= n <= MAX_KENO_NUMBER:
                tgt[n - 1] = 1
        Y.append(tgt)

    return np.array(X_seq), np.array(X_glob), np.array(X_spec), np.array(Y)


# ==============================================================================
# MODEL ARCHITECTURE WITH ATTENTION
# ==============================================================================

def build_attention_model() -> keras.Model:
    """
    Build multi-input model with attention mechanism.
    - Sequence branch: LSTM + Multi-head attention
    - Global branch: Dense network for advanced features
    - Specific branch: Dense for draw-specific features
    """
    # Sequence branch
    inp_seq = keras.Input(shape=(SEQUENCE_LENGTH, MAX_KENO_NUMBER), name="seq")
    x_seq = layers.Masking(mask_value=0.0)(inp_seq)
    x_seq = layers.LSTM(128, return_sequences=True)(x_seq)
    x_seq = layers.LayerNormalization()(x_seq)

    # Multi-head self-attention
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x_seq, x_seq)
    x_seq = layers.Add()([x_seq, attn])
    x_seq = layers.LayerNormalization()(x_seq)
    x_seq = layers.GlobalAveragePooling1D()(x_seq)

    # Global branch
    inp_glob = keras.Input(shape=(MAX_KENO_NUMBER * 12,), name="glob")
    x_glob = layers.Dense(256, activation="relu")(inp_glob)
    x_glob = layers.BatchNormalization()(x_glob)
    x_glob = layers.Dropout(0.3)(x_glob)
    x_glob = layers.Dense(128, activation="relu")(x_glob)

    # Specific branch
    inp_spec = keras.Input(shape=(10,), name="spec")
    x_spec = layers.Dense(64, activation="relu")(inp_spec)
    x_spec = layers.Dropout(0.2)(x_spec)

    # Combine branches
    combined = layers.concatenate([x_seq, x_glob, x_spec])
    x = layers.Dense(256, activation="relu")(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Output
    out = layers.Dense(MAX_KENO_NUMBER, activation="sigmoid")(x)

    model = keras.Model(inputs=[inp_seq, inp_glob, inp_spec], outputs=out)
    model.compile(
        loss=focal_loss_fixed,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    return model


# ==============================================================================
# CROSS-VALIDATION FOR TIME SERIES
# ==============================================================================

def time_series_cross_validation(model_fn, X_seq, X_glob, X_spec, Y, n_folds: int = CV_FOLDS):
    """
    Perform time-series cross-validation.
    Returns average metrics across folds.
    """
    if len(X_seq) < CV_MIN_SAMPLES:
        print(f"Not enough samples for CV ({len(X_seq)} < {CV_MIN_SAMPLES}), skipping...")
        return None

    tscv = TimeSeriesSplit(n_splits=n_folds)
    metrics = {"precision_at_20": [], "recall_at_20": [], "hit_rate": []}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
        print(f"Training fold {fold + 1}/{n_folds}...")

        # Create new model for this fold
        model = model_fn()

        # Train
        model.fit(
            [X_seq[train_idx], X_glob[train_idx], X_spec[train_idx]],
            Y[train_idx],
            validation_data=(
                [X_seq[val_idx], X_glob[val_idx], X_spec[val_idx]],
                Y[val_idx]
            ),
            epochs=50,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                )
            ]
        )

        # Evaluate
        probs = model.predict([X_seq[val_idx], X_glob[val_idx], X_spec[val_idx]], verbose=0)

        # Calculate metrics
        for i in range(len(probs)):
            pred_top20 = set(np.argsort(probs[i])[::-1][:20] + 1)
            actual = set(np.where(Y[i] == 1)[0] + 1)

            hits = len(pred_top20 & actual)
            metrics["hit_rate"].append(hits / 20.0)
            metrics["precision_at_20"].append(hits / max(1, len(pred_top20)))
            metrics["recall_at_20"].append(hits / max(1, len(actual)))

    # Print average metrics
    print("\nCross-validation results:")
    for metric, values in metrics.items():
        if values:
            print(f"  {metric}: {np.mean(values):.4f}")

    return metrics


# ==============================================================================
# SET OPTIMIZATION
# ==============================================================================

def top_k_core(probs: np.ndarray, k: int = 20) -> List[int]:
    """Get top K numbers by probability."""
    idx = np.argsort(probs)[::-1][:k]
    return [int(i + 1) for i in idx]


def build_sets_from_core20(
    core20: List[int],
    probs: np.ndarray,
    corr: np.ndarray,
    sizes: Tuple = (3, 5, 8),
    num_candidates: int = 30,
    temp: float = 1.0,
    mu: float = 0.06,
    seed: int = 2468
) -> Dict[str, List[int]]:
    """
    Build optimized subsets from core20 using correlation-aware selection.
    Uses simulated annealing-style probability adjustment.
    """
    rng = np.random.default_rng(seed)

    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 1e-9, 1.0)

    core = [int(x) for x in core20 if 1 <= int(x) <= MAX_KENO_NUMBER]
    core = list(dict.fromkeys(core))  # Unique, preserve order

    def weight_for(cand: int, chosen: List[int]) -> float:
        """Calculate adjusted weight for a candidate number."""
        w = p[cand - 1] ** (1.0 / max(1e-6, float(temp)))
        if chosen:
            pen = 0.0
            for ex in chosen:
                pen += float(corr[ex, cand])
            w *= float(np.exp(-mu * pen))
        return max(1e-12, float(w))

    def make_candidate(k: int) -> List[int]:
        """Generate one candidate set of size k."""
        chosen = []
        available = core[:]
        while len(chosen) < k and available:
            weights = np.array([weight_for(c, chosen) for c in available], dtype=float)
            s = float(weights.sum())
            if s <= 0:
                pick = int(available[0])
            else:
                probs_w = weights / s
                pick = int(rng.choice(available, p=probs_w))
            chosen.append(pick)
            available.remove(pick)
        return sorted(chosen)

    def set_score(nums: List[int]) -> float:
        """Score a set based on probabilities and correlation penalty."""
        nums = [int(x) for x in nums]
        base = float(sum(p[n - 1] for n in nums))
        pen = 0.0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pen += float(corr[nums[i], nums[j]])
        return base - mu * pen

    out = {}
    for k in sizes:
        best = None
        best_sc = -1e18
        for _ in range(num_candidates):
            cand = make_candidate(int(k))
            sc = set_score(cand)
            if sc > best_sc:
                best_sc = sc
                best = cand
        out[f"pick_{int(k)}"] = best if best is not None else sorted(core[:int(k)])

    return out


def tune_builder_params(
    probs_matrix: np.ndarray,
    y_matrix: np.ndarray,
    corr: np.ndarray,
    sizes: Tuple = (3, 5, 8),
    core_n: int = 20
) -> Dict[int, Dict[str, float]]:
    """
    Tune set builder parameters using backtesting on recent samples.
    Returns dict mapping size to optimal (temp, mu) params.
    """
    params = {}
    N = probs_matrix.shape[0]
    if N == 0:
        for s in sizes:
            params[int(s)] = {"temp": 1.0, "mu": 0.06}
        return params

    for s in sizes:
        best = {"temp": 1.0, "mu": 0.06}
        best_hits = -1e18

        for temp in TUNE_TEMPS:
            for mu in TUNE_MUS:
                total = 0.0
                for i in range(N):
                    probs = probs_matrix[i]
                    core20 = top_k_core(probs, core_n)
                    picks = build_sets_from_core20(
                        core20,
                        probs,
                        corr,
                        sizes=(int(s),),
                        num_candidates=12,
                        temp=float(temp),
                        mu=float(mu),
                        seed=RNG_SEED + i + int(s) * 1000,
                    )[f"pick_{int(s)}"]

                    y = y_matrix[i]
                    hits = 0
                    for n in picks:
                        if 1 <= n <= MAX_KENO_NUMBER:
                            hits += int(y[n - 1])
                    total += hits

                avg_hits = total / float(N)
                if avg_hits > best_hits:
                    best_hits = avg_hits
                    best = {"temp": float(temp), "mu": float(mu)}

        params[int(s)] = best

    return params


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_prediction_model_workflow():
    """Main execution for smart model."""
    print("=" * 60)
    print(f"--- SMART MODEL v2.0 ({PREDICTIONS_SAVE_FILE}) ---")
    print("=" * 60)

    game_ids, raw, n = load_data(DATA_FILE_PATH)
    if n == 0 or raw.size == 0:
        print("No data loaded.")
        return

    print(f"Loaded {n} draws from {DATA_FILE_PATH}")

    data = np.copy(raw)
    data[(data < 1) | (data > MAX_KENO_NUMBER)] = 0

    # Build relation matrix
    corr = build_relation_matrix(data)
    print("Built correlation matrix")

    # Create dataset
    X_seq, X_glob, X_spec, Y = create_dataset(data)
    if len(X_seq) == 0:
        print("Not enough sequences.")
        return

    print(f"Dataset size: {len(X_seq)} samples")

    # Load or create model
    model = None
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            with tf.keras.utils.custom_object_scope({
                "focal_loss_fixed": focal_loss_fixed,
                "precision_recall_loss": precision_recall_loss
            }):
                model = keras.models.load_model(MODEL_SAVE_PATH)
                print("Loaded existing model")
        except Exception as e:
            print(f"Could not load model: {e}")

    # Determine if full retrain needed
    full_retrain = (len(game_ids) > 0 and int(game_ids[-1]) % FULL_RETRAIN_INTERVAL == 0)
    if model is None:
        print("Building NEW model (force full retrain)")
        model = build_attention_model()
        full_retrain = True

    if full_retrain:
        print("Mode: FULL RETRAIN")
        split_idx = max(0, len(X_seq) - 2500)
        model.fit(
            [X_seq[split_idx:], X_glob[split_idx:], X_spec[split_idx:]],
            Y[split_idx:],
            validation_split=VALIDATION_SPLIT,
            epochs=FULL_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=20, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
                )
            ]
        )
    else:
        print("Mode: MINI UPDATE")
        split_idx = max(0, len(X_seq) - 250)
        model.fit(
            [X_seq[split_idx:], X_glob[split_idx:], X_spec[split_idx:]],
            Y[split_idx:],
            epochs=MINI_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )

    model.save(MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}")

    # Optional: Run cross-validation if enough data
    # Uncomment to enable:
    # cv_metrics = time_series_cross_validation(
    #     build_attention_model, X_seq, X_glob, X_spec, Y
    # )

    # Backtest for parameter tuning
    tune_n = min(TUNE_SAMPLES, len(X_seq))
    if tune_n >= 10:
        Xt_seq = X_seq[-tune_n:]
        Xt_glob = X_glob[-tune_n:]
        Xt_spec = X_spec[-tune_n:]
        Yt = Y[-tune_n:]

        probs_hist = model.predict([Xt_seq, Xt_glob, Xt_spec], verbose=0)
        params = tune_builder_params(probs_hist, Yt, corr, sizes=SET_SIZES, core_n=CORE_N)
    else:
        params = {int(s): {"temp": 1.0, "mu": 0.06} for s in SET_SIZES}

    # Predict next
    p_seq = compute_sequence_features(data[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, MAX_KENO_NUMBER)
    p_hist = compute_advanced_features(data[-LONG_LOOK:]).reshape(1, -1)
    p_spec = compute_draw_specific_features(data[-1]).reshape(1, -1)

    probs = model.predict([p_seq, p_hist, p_spec], verbose=0)[0]

    # Apply brain bias
    bias = load_brain_bias()
    for i in range(MAX_KENO_NUMBER):
        probs[i] *= float(bias[i + 1])

    core_20 = top_k_core(probs, CORE_N)

    # Build optimized sets
    predicted_sets = {}
    for s in SET_SIZES:
        t = params[int(s)]["temp"]
        mu = params[int(s)]["mu"]
        predicted_sets.update(
            build_sets_from_core20(
                core_20,
                probs,
                corr,
                sizes=(int(s),),
                num_candidates=CANDIDATES_PER_SIZE,
                temp=float(t),
                mu=float(mu),
                seed=RNG_SEED + int(s) * 999,
            )
        )

    predicted_sets["pick_20"] = sorted([int(x) for x in core_20])
    for s in EXTRA_STYLE_SIZES:
        predicted_sets[f"pick_{int(s)}"] = sorted([int(x) for x in core_20[:int(s)]])

    output = {
        "core_20": [int(x) for x in core_20],
        "probs": [float(x) for x in probs.tolist()],
        "predicted_sets": predicted_sets,
        "meta": {
            "model": "Smart_v2",
            "tuned_params": {str(k): v for k, v in params.items()},
            "samples": len(X_seq),
        },
    }

    with open(PREDICTIONS_SAVE_FILE, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Saved {PREDICTIONS_SAVE_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    run_prediction_model_workflow()
