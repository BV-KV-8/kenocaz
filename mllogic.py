"""
Keno Predictor Logic - Legacy Model (Upgraded)
====================================================

Advanced ML model with enhanced feature engineering:
- Time since last appearance for each number
- Frequency of consecutive appearances
- Positional analysis features
- Lagged features from previous N draws
- Statistical properties (variance, skewness, kurtosis)
- Interaction features between numbers

Author: Keno Prediction System
Version: 2.0
"""

import os
import csv
import json
import random
import warnings
from collections import Counter, defaultdict
from typing import Tuple, List, Dict, Optional

import numpy as np
import scipy.stats as stats

import tensorflow as tf
from tensorflow import keras
from keras import layers

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

print("=" * 60)
print("Keno Predictor Logic (Legacy v2.0) - Enhanced Features")
print("=" * 60)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MAX_KENO_NUMBER = 80
DATA_FILE_PATH = "games.csv"
MODEL_SAVE_PATH = "keno_prediction_model.keras"
PREDICTIONS_SAVE_FILE = "last_predictions.json"
BRAINSTATE_FILE = "brainstate.json"

# Lookback windows for feature extraction
SHORT_TERM_LOOKBACK = 10
MEDIUM_TERM_LOOKBACK = 30
LONG_TERM_LOOKBACK = 50
VERY_LONG_LOOKBACK = 100

# Sequence length for LSTM-style features
SEQUENCE_LENGTH = 10

# Training configuration
FULL_RETRAIN_INTERVAL = 12
MINI_EPOCHS = 5
FULL_EPOCHS = 80
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

# Model architecture
HIDDEN_UNITS = [512, 256, 128]
DROPOUT_RATES = [0.4, 0.3, 0.2]

# Play styles for predictions
PLAY_STYLES = [3, 5, 6, 7, 8, 10, 15, 20]

# Random seed for reproducibility
RNG_SEED = 42

# ==============================================================================
# CUSTOM LOSS FUNCTIONS
# ==============================================================================

@tf.keras.utils.register_keras_serializable()
def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples.
    """
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
def weighted_binary_crossentropy(y_true, y_pred, pos_weight=3.0):
    """
    Weighted binary cross-entropy to handle class imbalance.
    Positive class (numbers that appear) gets higher weight.
    """
    y_true = tf.cast(y_true, tf.float32)
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    bce = -(y_true * pos_weight * tf.math.log(y_pred) +
             (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(bce, axis=-1)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_brain_bias() -> List[float]:
    """
    Load global bias weights from brainstate file.
    Returns list of 81 floats (index 0 unused).
    """
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


def load_keno_data_with_ids(filepath: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load Keno data from canonical CSV format.
    Format: draw_id, timestamp, n1, n2, ..., n20

    Returns:
        ids: numpy array of draw IDs
        draws: numpy array of shape (N, 20) containing the 20 numbers per draw
        count: number of valid draws loaded
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
# ADVANCED FEATURE ENGINEERING
# ==============================================================================

def compute_time_since_last_appearance(history: np.ndarray, lookback: int = 100) -> np.ndarray:
    """
    Compute games since each number last appeared.
    Returns array of shape (80,) with normalized gap values.
    """
    gaps = np.zeros(MAX_KENO_NUMBER, dtype=float)
    last_seen = {}

    recent = history[-lookback:] if len(history) >= lookback else history

    for i, draw in enumerate(reversed(recent)):
        for n in draw:
            if 1 <= n <= MAX_KENO_NUMBER:
                if n not in last_seen:
                    gaps[n - 1] = i
                    last_seen[n] = i

    # For numbers never seen, use maximum gap
    for n in range(1, MAX_KENO_NUMBER + 1):
        if n not in last_seen:
            gaps[n - 1] = len(recent)

    # Normalize
    if len(recent) > 0:
        gaps = gaps / len(recent)
    return gaps


def compute_consecutive_appearance_frequency(history: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Compute frequency of consecutive appearances for each number.
    A number appearing in back-to-back games increments its consecutive count.
    """
    consecutive_counts = np.zeros(MAX_KENO_NUMBER, dtype=float)
    total_opportunities = 0

    recent = history[-lookback:] if len(history) >= lookback else history

    if len(recent) < 2:
        return consecutive_counts

    for i in range(1, len(recent)):
        total_opportunities += 1
        prev_set = set(recent[i - 1])
        curr_set = set(recent[i])

        for n in curr_set:
            if 1 <= n <= MAX_KENO_NUMBER and n in prev_set:
                consecutive_counts[n - 1] += 1

    # Normalize
    if total_opportunities > 0:
        consecutive_counts = consecutive_counts / total_opportunities
    return consecutive_counts


def compute_positional_features(history: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Compute positional analysis features.
    Tracks where in the draw sequence (1-20) each number appears.
    """
    position_features = np.zeros((MAX_KENO_NUMBER, 20), dtype=float)

    recent = history[-lookback:] if len(history) >= lookback else history

    for draw in recent:
        for pos, n in enumerate(draw):
            if 1 <= n <= MAX_KENO_NUMBER:
                position_features[n - 1, pos] += 1

    # Normalize by number of draws
    if len(recent) > 0:
        position_features = position_features / len(recent)

    return position_features.flatten()


def compute_lagged_features(history: np.ndarray, lag_windows: List[int] = None) -> np.ndarray:
    """
    Compute lagged features - actual occurrence indicators from previous N draws.
    Creates binary features for each number at each lag window.
    """
    if lag_windows is None:
        lag_windows = [1, 2, 3, 5, 10]

    features = []

    for lag in lag_windows:
        if len(history) >= lag:
            recent_draws = history[-lag:]
            # Flatten the draws and create binary indicator
            lagged = np.zeros(MAX_KENO_NUMBER, dtype=float)
            for draw in recent_draws:
                for n in draw:
                    if 1 <= n <= MAX_KENO_NUMBER:
                        lagged[n - 1] = 1.0
            features.append(lagged)
        else:
            features.append(np.zeros(MAX_KENO_NUMBER, dtype=float))

    return np.concatenate(features)


def compute_statistical_features(history: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Compute statistical properties of each number's appearance pattern.
    Includes variance, skewness, kurtosis of inter-arrival times.
    """
    stats_features = np.zeros((MAX_KENO_NUMBER, 4), dtype=float)

    recent = history[-lookback:] if len(history) >= lookback else history

    for n in range(1, MAX_KENO_NUMBER + 1):
        # Track positions where this number appears
        appearances = []
        for i, draw in enumerate(recent):
            if n in draw:
                appearances.append(i)

        if len(appearances) > 1:
            # Inter-arrival times
            intervals = np.diff(appearances)

            # Variance (normalized)
            stats_features[n - 1, 0] = np.var(intervals) / max(1, lookback)

            # Skewness
            if len(intervals) >= 3:
                try:
                    stats_features[n - 1, 1] = stats.skew(intervals)
                except:
                    stats_features[n - 1, 1] = 0.0

            # Kurtosis
            if len(intervals) >= 4:
                try:
                    stats_features[n - 1, 2] = stats.kurtosis(intervals)
                except:
                    stats_features[n - 1, 2] = 0.0

        # Mean position in draw when appears (0-1)
        if len(appearances) > 0:
            positions_in_draw = []
            for draw in recent:
                if n in draw:
                    positions_in_draw.append(draw.tolist().index(n) / 20.0)
            if positions_in_draw:
                stats_features[n - 1, 3] = np.mean(positions_in_draw)

    return stats_features.flatten()


def compute_interaction_features(history: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Compute interaction features between number pairs.
    Uses co-occurrence patterns to create pairwise affinity scores.
    """
    interaction_features = np.zeros(MAX_KENO_NUMBER, dtype=float)

    recent = history[-lookback:] if len(history) >= lookback else history

    # Count total co-occurrences for each number
    cooccurrence_counts = defaultdict(int)

    for draw in recent:
        for n in draw:
            if 1 <= n <= MAX_KENO_NUMBER:
                cooccurrence_counts[n] += len([x for x in draw if x != n])

    # Normalize and create feature
    max_count = max(cooccurrence_counts.values()) if cooccurrence_counts else 1
    for n in range(1, MAX_KENO_NUMBER + 1):
        interaction_features[n - 1] = cooccurrence_counts.get(n, 0) / max_count

    return interaction_features


def calc_enhanced_global_features(history: np.ndarray) -> np.ndarray:
    """
    Combine all advanced features into a single feature vector.
    Returns flattened feature array.
    """
    if len(history) == 0:
        # Return zeros with correct dimension
        dim = (MAX_KENO_NUMBER * 7 +           # Basic features + consecutive
                MAX_KENO_NUMBER * 20 +          # Positional features
                MAX_KENO_NUMBER * 5 +           # Lagged features (5 lags)
                MAX_KENO_NUMBER * 4 +           # Statistical features
                MAX_KENO_NUMBER)                # Interaction features
        return np.zeros(dim, dtype=float)

    feature_components = []

    # 1. Basic frequency features (6 per number)
    feats = np.zeros((MAX_KENO_NUMBER, 7), dtype=float)

    sh = history[-SHORT_TERM_LOOKBACK:].flatten() if len(history) >= SHORT_TERM_LOOKBACK else history.flatten()
    mh = history[-MEDIUM_TERM_LOOKBACK:].flatten() if len(history) >= MEDIUM_TERM_LOOKBACK else history.flatten()
    lh = history[-LONG_TERM_LOOKBACK:].flatten() if len(history) >= LONG_TERM_LOOKBACK else history.flatten()

    sc = Counter(sh)
    mc = Counter(mh)
    lc = Counter(lh)

    # Time since last appearance
    gaps = compute_time_since_last_appearance(history, LONG_TERM_LOOKBACK)
    # Consecutive appearance frequency
    consecutive = compute_consecutive_appearance_frequency(history, MEDIUM_TERM_LOOKBACK)

    for n in range(1, MAX_KENO_NUMBER + 1):
        idx = n - 1
        feats[idx, 0] = sc.get(n, 0) / max(1, SHORT_TERM_LOOKBACK)
        feats[idx, 1] = mc.get(n, 0) / max(1, MEDIUM_TERM_LOOKBACK)
        feats[idx, 2] = lc.get(n, 0) / max(1, LONG_TERM_LOOKBACK)
        feats[idx, 3] = gaps[idx]
        feats[idx, 4] = 1.0 if n % 2 == 0 else 0.0  # Even
        feats[idx, 5] = 1.0 if n <= 40 else 0.0  # First half
        feats[idx, 6] = consecutive[idx]  # Consecutive frequency

    feature_components.append(feats.flatten())

    # 2. Positional features (20 per number)
    pos_feats = compute_positional_features(history, MEDIUM_TERM_LOOKBACK)
    feature_components.append(pos_feats)

    # 3. Lagged features (5 per number - 5 lags)
    lag_feats = compute_lagged_features(history, [1, 2, 3, 5, 10])
    feature_components.append(lag_feats)

    # 4. Statistical features (4 per number)
    stat_feats = compute_statistical_features(history, MEDIUM_TERM_LOOKBACK)
    feature_components.append(stat_feats)

    # 5. Interaction features (1 per number)
    inter_feats = compute_interaction_features(history, MEDIUM_TERM_LOOKBACK)
    feature_components.append(inter_feats)

    return np.concatenate(feature_components)


def calc_sequence_features(history: np.ndarray, seq_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    Create sequence features for LSTM/attention processing.
    Returns binary matrix of shape (seq_length, 80).
    """
    if len(history) < seq_length:
        # Pad with zeros if not enough history
        padded = np.zeros((seq_length, 20), dtype=int)
        start = seq_length - len(history)
        for i, draw in enumerate(history):
            for j, n in enumerate(draw):
                if 1 <= n <= MAX_KENO_NUMBER:
                    padded[start + i, j] = n
        history = padded

    recent = history[-seq_length:]
    seq_matrix = np.zeros((seq_length, MAX_KENO_NUMBER), dtype=float)

    for i, draw in enumerate(recent):
        for n in draw:
            if 1 <= n <= MAX_KENO_NUMBER:
                seq_matrix[i, n - 1] = 1.0

    return seq_matrix


# ==============================================================================
# DATASET CREATION
# ==============================================================================

def create_enhanced_dataset(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training dataset with enhanced features.
    Returns X (features) and Y (targets).
    """
    lookback = max(SHORT_TERM_LOOKBACK, MEDIUM_TERM_LOOKBACK, LONG_TERM_LOOKBACK, SEQUENCE_LENGTH)

    if len(data) < lookback + 2:
        return np.array([]), np.array([])

    X, Y = [], []

    for i in range(lookback, len(data) - 1):
        # Global features from history
        feats = calc_enhanced_global_features(data[i - lookback : i])
        X.append(feats)

        # Target: binary vector for next draw
        tgt = np.zeros(MAX_KENO_NUMBER, dtype=int)
        for n in data[i + 1]:
            if 1 <= n <= MAX_KENO_NUMBER:
                tgt[n - 1] = 1
        Y.append(tgt)

    return np.array(X), np.array(Y)


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

def build_enhanced_model(input_dim: int) -> keras.Model:
    """
    Build enhanced neural network with:
    - Deeper architecture
    - Regularization (dropout, batch normalization)
    - Residual connections
    """
    inp = keras.Input(shape=(input_dim,), name="features")

    # First block
    x = layers.Dense(HIDDEN_UNITS[0], activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATES[0])(x)

    # Second block
    x = layers.Dense(HIDDEN_UNITS[1], activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATES[1])(x)

    # Third block
    x = layers.Dense(HIDDEN_UNITS[2], activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATES[2])(x)

    # Output block
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    out = layers.Dense(MAX_KENO_NUMBER, activation="sigmoid")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        loss=focal_loss_fixed,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    return model


# ==============================================================================
# PREDICTION
# ==============================================================================

def predict_with_model(
    model: keras.Model,
    data: np.ndarray,
    bias: List[float]
) -> np.ndarray:
    """
    Generate predictions using trained model.
    """
    lookback = max(SHORT_TERM_LOOKBACK, MEDIUM_TERM_LOOKBACK,
                  LONG_TERM_LOOKBACK, SEQUENCE_LENGTH)

    # Global features
    feats = calc_enhanced_global_features(data[-lookback:]).reshape(1, -1)
    probs = model.predict(feats, verbose=0)[0]

    # Apply brain bias
    for i in range(MAX_KENO_NUMBER):
        probs[i] *= float(bias[i + 1])

    return probs


def generate_prediction_sets(probs: np.ndarray) -> Dict[str, List[int]]:
    """
    Generate prediction sets for various play styles.
    """
    top = np.argsort(probs)[::-1]
    predicted_sets = {}

    for picks in PLAY_STYLES:
        predicted_sets[f"pick_{picks}"] = sorted([int(i + 1) for i in top[:picks]])

    return predicted_sets


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run():
    """Main execution function for the legacy model."""
    print("=" * 60)
    print("Starting Legacy Model Training & Prediction")
    print("=" * 60)

    # Load data
    ids, draws, n = load_keno_data_with_ids(DATA_FILE_PATH)
    if n == 0 or draws.size == 0:
        print("No data loaded.")
        return

    print(f"Loaded {n} draws from {DATA_FILE_PATH}")

    # Clean data
    data = np.copy(draws)
    data[(data < 1) | (data > MAX_KENO_NUMBER)] = 0

    # Create dataset
    X, Y = create_enhanced_dataset(data)
    if len(X) == 0:
        print("Not enough data for training.")
        return

    print(f"Dataset size: {len(X)} samples")
    print(f"Feature dimension: {X.shape[1]}")

    # Split for validation
    split_idx = int(len(X) * (1 - VALIDATION_SPLIT))
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]

    # Load or build model
    model = None

    if os.path.exists(MODEL_SAVE_PATH):
        try:
            with tf.keras.utils.custom_object_scope({
                "focal_loss_fixed": focal_loss_fixed
            }):
                model = keras.models.load_model(MODEL_SAVE_PATH)
                print("Loaded existing model")
        except Exception as e:
            print(f"Could not load model: {e}")

    # Build model if needed
    full_retrain = (len(ids) > 0 and int(ids[-1]) % FULL_RETRAIN_INTERVAL == 0)
    if model is None or full_retrain:
        print("Training new model...")
        model = build_enhanced_model(X.shape[1])
        model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=FULL_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6
                )
            ]
        )
        model.save(MODEL_SAVE_PATH)
        print(f"Saved model to {MODEL_SAVE_PATH}")

    # Generate predictions
    bias = load_brain_bias()
    probs = predict_with_model(model, data, bias)

    # Generate sets
    predicted_sets = generate_prediction_sets(probs)

    # Save predictions
    output = {
        "probs": [float(x) for x in probs.tolist()],
        "predicted_sets": predicted_sets,
        "meta": {
            "model": "Legacy_v2",
            "samples": len(X),
            "features": X.shape[1],
            "last_draw_id": int(ids[-1]) if len(ids) > 0 else None
        }
    }

    with open(PREDICTIONS_SAVE_FILE, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Saved predictions to {PREDICTIONS_SAVE_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    run()
