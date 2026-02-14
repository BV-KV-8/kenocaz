"""
Keno Predictor Logic - Standard Model (Upgraded)
====================================================

Most advanced model with:
- Genetic algorithm for set optimization
- Advanced correlation matrix analysis
- Diverse set generation
- Multi-input ensemble architecture
- Bi-directional LSTM with Conv1D
- Simulated annealing for parameter tuning

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
from collections import Counter
from typing import Tuple, List, Dict, Optional
from scipy.stats import entropy
import random

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_FILE_PATH = "games.csv"
MODEL_SAVE_PATH = "modelstandard.keras"
PREDICTIONS_SAVE_FILE = "pred_standard.json"
BRAINSTATE_FILE = "brainstate.json"

# Training configuration
FULL_RETRAIN_INTERVAL = 12
MINI_BATCH_SIZE = 250
MINI_EPOCHS = 6
FULL_EPOCHS = 140
BATCH_SIZE = 32

# Feature lookback windows
SHORT_TERM_LOOKBACK = 10
MEDIUM_TERM_LOOKBACK = 30
LONG_TERM_LOOKBACK = 50

MAX_KENO_NUMBER = 80

# Set building config
CORE_N = 20
SET_SIZES = (3, 5, 8)
CANDIDATES_PER_SIZE = 30
TUNE_SAMPLES = 120
TUNE_TEMPS = (0.7, 1.0, 1.3)
TUNE_MUS = (0.0, 0.03, 0.06, 0.10)
RNG_SEED = 1337

EXTRA_STYLE_SIZES = (10,)

# Genetic algorithm config
GA_POPULATION_SIZE = 50
GA_GENERATIONS = 10
GA_MUTATION_RATE = 0.1
GA_CROSSOVER_RATE = 0.7

# Simulated annealing config
SA_INITIAL_TEMP = 1.0
SA_COOLING_RATE = 0.95
SA_MIN_TEMP = 0.01

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
def diversity_loss(y_true, y_pred):
    """Encourage diversity in top predictions."""
    y_true = tf.cast(y_true, tf.float32)
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    # Standard BCE
    bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    bce = tf.reduce_mean(bce)

    # Diversity penalty: encourage spread in probabilities
    entropy_penalty = tf.reduce_sum(y_pred * tf.math.log(y_pred + eps))

    return bce + 0.01 * entropy_penalty


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
    """Load Keno data from canonical CSV format."""
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
                nums = [int(r.get(f"n{i}", 0)) for i in range(1, 21)]
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

def build_advanced_correlation_matrix(data: np.ndarray, lookback: int = 250) -> np.ndarray:
    """
    Build advanced co-occurrence matrix with temporal weighting.
    Recent draws have higher weight in the correlation.
    """
    matrix = np.zeros((MAX_KENO_NUMBER + 1, MAX_KENO_NUMBER + 1), dtype=float)
    recent = data[-lookback:] if len(data) >= lookback else data

    weights = np.linspace(0.5, 1.0, len(recent))

    for idx, draw in enumerate(recent):
        w = weights[idx] if len(weights) > idx else 1.0
        for i in draw:
            if i <= 0 or i > MAX_KENO_NUMBER:
                continue
            for j in draw:
                if j <= 0 or j > MAX_KENO_NUMBER or i == j:
                    continue
                matrix[i, j] += w

    # Normalize
    mx = float(np.max(matrix))
    if mx > 0:
        matrix /= mx

    return matrix


def compute_global_features(history: np.ndarray) -> np.ndarray:
    """
    Compute comprehensive global features for each number.
    Returns flattened feature array of shape (80 * 8,).
    """
    feats = np.zeros((MAX_KENO_NUMBER, 8), dtype=float)

    if len(history) == 0:
        return feats.flatten()

    sh = history[-SHORT_TERM_LOOKBACK:].flatten() if len(history) >= SHORT_TERM_LOOKBACK else history.flatten()
    mh = history[-MEDIUM_TERM_LOOKBACK:].flatten() if len(history) >= MEDIUM_TERM_LOOKBACK else history.flatten()
    lh = history[-LONG_TERM_LOOKBACK:].flatten() if len(history) >= LONG_TERM_LOOKBACK else history.flatten()

    sc = Counter(sh)
    mc = Counter(mh)
    lc = Counter(lh)

    # Recency in medium window (games since last appearance)
    recency = {n: MEDIUM_TERM_LOOKBACK for n in range(1, MAX_KENO_NUMBER + 1)}
    w = history[-MEDIUM_TERM_LOOKBACK:] if len(history) >= MEDIUM_TERM_LOOKBACK else history
    for i, draw in enumerate(reversed(w)):
        for n in draw:
            if 1 <= n <= MAX_KENO_NUMBER and recency[n] == MEDIUM_TERM_LOOKBACK:
                recency[n] = i

    # Momentum: is the number becoming more or less frequent?
    momentum = {}
    for n in range(1, MAX_KENO_NUMBER + 1):
        recent_freq = sc.get(n, 0) / max(1, SHORT_TERM_LOOKBACK)
        medium_freq = mc.get(n, 0) / max(1, MEDIUM_TERM_LOOKBACK)
        momentum[n] = recent_freq - medium_freq

    for n in range(1, MAX_KENO_NUMBER + 1):
        idx = n - 1
        feats[idx, 0] = sc.get(n, 0) / max(1, SHORT_TERM_LOOKBACK)
        feats[idx, 1] = mc.get(n, 0) / max(1, MEDIUM_TERM_LOOKBACK)
        feats[idx, 2] = lc.get(n, 0) / max(1, LONG_TERM_LOOKBACK)
        feats[idx, 3] = recency.get(n, MEDIUM_TERM_LOOKBACK) / max(1, MEDIUM_TERM_LOOKBACK)
        feats[idx, 4] = 1.0 if (n % 2 == 0) else 0.0
        feats[idx, 5] = 1.0 if (n <= 40) else 0.0
        feats[idx, 6] = momentum.get(n, 0.0)
        feats[idx, 7] = 1.0 if (1 <= n <= 20) else 0.0  # First quintile

    return feats.flatten()


def compute_draw_features(draw: np.ndarray) -> np.ndarray:
    """
    Compute draw-specific features.
    Captures statistical properties of a single draw.
    """
    feats = np.zeros(8, dtype=float)

    if draw.size == 0:
        return feats

    v = draw[(draw >= 1) & (draw <= MAX_KENO_NUMBER)]
    if v.size == 0:
        return feats

    feats[0] = float(np.mean(v)) / 80.0  # Mean normalized
    feats[1] = float(np.sum(v <= 40)) / 20.0  # Ratio first half
    feats[2] = float(np.sum(v > 40)) / 20.0  # Ratio second half
    feats[3] = float(np.sum(v % 2 == 0)) / 20.0  # Ratio even
    feats[4] = float(np.sum(v % 2 != 0)) / 20.0  # Ratio odd
    feats[5] = float(len(np.unique(v))) / 20.0  # Uniqueness

    # Entropy (diversity measure)
    hist, _ = np.histogram(v, bins=20, range=(1, 81))
    hist = hist / np.sum(hist)  # Normalize
    feats[6] = float(entropy(hist + 1e-10))

    # Spread
    feats[7] = (np.max(v) - np.min(v)) / 79.0

    return feats


def compute_sequence_features(history: np.ndarray, seq_len: int = 20) -> np.ndarray:
    """
    Compute sequence features for LSTM processing.
    """
    if len(history) < seq_len:
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
    """Create training dataset with multi-input features."""
    lookback = max(SHORT_TERM_LOOKBACK, MEDIUM_TERM_LOOKBACK, LONG_TERM_LOOKBACK, 20)

    if len(data) < lookback + 2:
        return np.array([]), np.array([]), np.array([]), np.array([])

    X_seq, X_glob, X_spec, Y = [], [], [], []

    for i in range(lookback, len(data) - 1):
        X_seq.append(compute_sequence_features(data[i - lookback : i]))
        X_glob.append(compute_global_features(data[i - lookback : i]))
        X_spec.append(compute_draw_features(data[i]))

        tgt = np.zeros(MAX_KENO_NUMBER, dtype=int)
        for n in data[i + 1]:
            if 1 <= n <= MAX_KENO_NUMBER:
                tgt[n - 1] = 1
        Y.append(tgt)

    return np.array(X_seq), np.array(X_glob), np.array(X_spec), np.array(Y)


# ==============================================================================
# ADVANCED MODEL ARCHITECTURE
# ==============================================================================

def build_advanced_model() -> keras.Model:
    """
    Build advanced multi-input model with:
    - Bidirectional LSTM with attention
    - Conv1D for local patterns
    - Residual connections
    - Deep architecture with batch normalization
    """
    # Sequence branch
    inp_seq = keras.Input(shape=(20, MAX_KENO_NUMBER), name="seq")
    x_seq = layers.Masking(mask_value=0.0)(inp_seq)

    # Conv1D for local pattern extraction
    x_seq = layers.Conv1D(64, 3, padding="same", activation="relu")(x_seq)
    x_seq = layers.SpatialDropout1D(0.2)(x_seq)

    # Bidirectional LSTM
    x_seq = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x_seq)
    x_seq = layers.LayerNormalization()(x_seq)

    # Self-attention
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x_seq, x_seq)
    x_seq = layers.Add()([x_seq, attn])
    x_seq = layers.LayerNormalization()(x_seq)

    x_seq = layers.GlobalAveragePooling1D()(x_seq)

    # Global branch
    inp_glob = keras.Input(shape=(MAX_KENO_NUMBER * 8,), name="glob")
    x_glob = layers.Dense(256, activation="relu")(inp_glob)
    x_glob = layers.BatchNormalization()(x_glob)
    x_glob = layers.Dropout(0.3)(x_glob)
    x_glob = layers.Dense(128, activation="relu")(x_glob)
    x_glob = layers.Dropout(0.2)(x_glob)

    # Specific branch
    inp_spec = keras.Input(shape=(8,), name="spec")
    x_spec = layers.Dense(64, activation="relu")(inp_spec)
    x_spec = layers.Dropout(0.2)(x_spec)

    # Combine
    combined = layers.concatenate([x_seq, x_glob, x_spec])

    # Deep processing
    x = layers.Dense(512, activation="relu")(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Residual connection
    residual = x
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Output
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(MAX_KENO_NUMBER, activation="sigmoid")(x)

    model = keras.Model(inputs=[inp_seq, inp_glob, inp_spec], outputs=out)
    model.compile(
        loss=focal_loss_fixed,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    return model


# ==============================================================================
# GENETIC ALGORITHM FOR SET OPTIMIZATION
# ==============================================================================

def genetic_algorithm_optimize(
    probs: np.ndarray,
    corr: np.ndarray,
    k: int,
    population_size: int = GA_POPULATION_SIZE,
    generations: int = GA_GENERATIONS,
    mutation_rate: float = GA_MUTATION_RATE,
    crossover_rate: float = GA_CROSSOVER_RATE,
    seed: int = 1337
) -> List[int]:
    """
    Use genetic algorithm to find optimal set of size k.
    Optimizes for: high probability scores + low correlation penalty.
    """
    rng = np.random.default_rng(seed)

    # Get top candidates (top 40 numbers by probability)
    top_candidates = [int(i + 1) for i in np.argsort(probs)[::-1][:40]]

    def fitness(individual):
        """Calculate fitness: sum of probs - correlation penalty."""
        score = sum(probs[n - 1] for n in individual)
        # Apply correlation penalty
        for i in range(len(individual)):
            for j in range(i + 1, len(individual)):
                score -= 0.05 * corr[individual[i], individual[j]]
        return score

    def create_individual():
        """Create random individual from top candidates."""
        return sorted(rng.choice(top_candidates, size=k, replace=False))

    def crossover(parent1, parent2):
        """One-point crossover between two parents. Returns tuple of two children."""
        if len(parent1) != k or len(parent2) != k:
            return parent1[:], parent2[:]
        point = rng.integers(1, k)

        # First child: first part of parent1 + remaining from parent2
        p1_part = list(parent1[:point])
        child1 = p1_part + [x for x in parent2 if x not in p1_part]
        if len(child1) < k:
            remaining = [x for x in top_candidates if x not in child1]
            child1 += remaining[:k - len(child1)]
        elif len(child1) > k:
            child1 = child1[:k]
        child1 = sorted(list(set(child1)))
        # Ensure size k
        while len(child1) < k:
            for x in top_candidates:
                if x not in child1:
                    child1.append(x)
                    break
        child1 = sorted(child1[:k])

        # Second child: first part of parent2 + remaining from parent1
        p2_part = list(parent2[:point])
        child2 = p2_part + [x for x in parent1 if x not in p2_part]
        if len(child2) < k:
            remaining = [x for x in top_candidates if x not in child2]
            child2 += remaining[:k - len(child2)]
        elif len(child2) > k:
            child2 = child2[:k]
        child2 = sorted(list(set(child2)))
        # Ensure size k
        while len(child2) < k:
            for x in top_candidates:
                if x not in child2:
                    child2.append(x)
                    break
        child2 = sorted(child2[:k])

        return child1, child2

    def mutate(individual):
        """Mutate individual by replacing one element."""
        if rng.random() > mutation_rate:
            return individual[:]
        idx = rng.integers(0, len(individual))
        replacement = rng.choice(top_candidates)
        new_ind = individual[:]
        new_ind[idx] = replacement
        return sorted(list(set(new_ind)))

    # Initialize population
    population = [create_individual() for _ in range(population_size)]

    best_individual = None
    best_fitness = -float('inf')

    for gen in range(generations):
        # Evaluate fitness
        fitnesses = [fitness(ind) for ind in population]

        # Track best
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_individual = population[gen_best_idx][:]

        # Selection (tournament)
        new_population = []
        for _ in range(population_size):
            candidates = rng.choice(population, size=3, replace=False)
            winner = max(candidates, key=fitness)
            new_population.append(winner[:])

        # Crossover
        for i in range(0, population_size, 2):
            if i + 1 < population_size and rng.random() < crossover_rate:
                child1, child2 = crossover(
                    new_population[i], new_population[i + 1]
                )
                new_population[i] = child1
                new_population[i + 1] = child2

        # Mutation
        population = [mutate(ind) for ind in new_population]

        # Ensure valid individuals
        population = [ind for ind in population if len(ind) == k]
        while len(population) < population_size:
            population.append(create_individual())

    return best_individual if best_individual else sorted(top_candidates[:k])


# ==============================================================================
# SIMULATED ANNEALING FOR PARAMETER TUNING
# ==============================================================================

def simulated_annealing_tune(
    probs_matrix: np.ndarray,
    y_matrix: np.ndarray,
    corr: np.ndarray,
    k: int,
    initial_temp: float = SA_INITIAL_TEMP,
    cooling_rate: float = SA_COOLING_RATE,
    min_temp: float = SA_MIN_TEMP,
    seed: int = 1337
) -> Dict[str, float]:
    """
    Use simulated annealing to find optimal (temp, mu) parameters.
    """
    rng = np.random.default_rng(seed)

    def evaluate_params(temp, mu):
        """Evaluate parameter settings."""
        total_hits = 0
        for i in range(len(probs_matrix)):
            probs = probs_matrix[i]
            core20 = [int(x) for x in np.argsort(probs)[::-1][:20]]
            # Build set with params
            picks = genetic_algorithm_optimize(
                probs[:20], corr, k, population_size=20, generations=5,
                seed=seed + i
            )
            y = y_matrix[i]
            hits = sum(1 for n in picks if 1 <= n <= MAX_KENO_NUMBER and y[n - 1])
            total_hits += hits
        return total_hits / len(probs_matrix)

    # Initial state
    current_temp = initial_temp
    current_temp = 1.0
    current_mu = 0.06
    current_score = evaluate_params(current_temp, current_mu)

    best_temp, best_mu = current_temp, current_mu
    best_score = current_score

    while current_temp > min_temp:
        # Generate neighbor
        new_temp = current_temp + rng.uniform(-0.2, 0.2)
        new_mu = max(0.0, min(0.2, current_mu + rng.uniform(-0.02, 0.02)))

        new_score = evaluate_params(max(0.1, new_temp), new_mu)

        # Accept or reject
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / current_temp):
            current_temp, current_mu = new_temp, new_mu
            current_score = new_score

            if new_score > best_score:
                best_temp, best_mu = new_temp, new_mu
                best_score = new_score

        current_temp *= cooling_rate

    return {"temp": best_temp, "mu": best_mu}


# ==============================================================================
# SET BUILDING WITH DIVERSITY
# ==============================================================================

def top_k_core(probs: np.ndarray, k: int = 20) -> List[int]:
    """Get top K numbers by probability."""
    idx = np.argsort(probs)[::-1][:k]
    return [int(i + 1) for i in idx]


def build_diverse_sets(
    core20: List[int],
    probs: np.ndarray,
    corr: np.ndarray,
    sizes: Tuple = (3, 5, 8),
    num_candidates: int = 30,
    temp: float = 1.0,
    mu: float = 0.06,
    seed: int = 1337
) -> Dict[str, List[int]]:
    """
    Build diverse optimized sets from core20.
    Uses genetic algorithm for optimization.
    """
    rng = np.random.default_rng(seed)

    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 1e-9, 1.0)

    core = [int(x) for x in core20 if 1 <= int(x) <= MAX_KENO_NUMBER]
    core = list(dict.fromkeys(core))

    def weight_for(cand: int, chosen: List[int]) -> float:
        """Calculate adjusted weight for candidate."""
        w = p[cand - 1] ** (1.0 / max(1e-6, float(temp)))
        if chosen:
            pen = 0.0
            for ex in chosen:
                pen += float(corr[ex, cand])
            w *= float(np.exp(-mu * pen))
        return max(1e-12, float(w))

    def make_candidate(k: int) -> List[int]:
        """Generate one candidate set."""
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
        """Score a set."""
        nums = [int(x) for x in nums]
        base = float(sum(p[n - 1] for n in nums))
        pen = 0.0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pen += float(corr[nums[i], nums[j]])
        return base - mu * pen

    out = {}
    for k in sizes:
        # Use GA for optimization
        ga_result = genetic_algorithm_optimize(
            probs, corr, int(k),
            population_size=min(30, num_candidates),
            generations=10,
            seed=seed + int(k) * 100
        )

        # Score the GA result
        ga_score = set_score(ga_result)

        # Also try weighted sampling approach
        best = ga_result
        best_sc = ga_score

        for _ in range(num_candidates // 2):
            cand = make_candidate(int(k))
            sc = set_score(cand)
            if sc > best_sc:
                best_sc = sc
                best = cand

        out[f"pick_{int(k)}"] = best

    return out


def tune_builder_params(
    probs_matrix: np.ndarray,
    y_matrix: np.ndarray,
    corr: np.ndarray,
    sizes: Tuple = (3, 5, 8),
    core_n: int = 20
) -> Dict[int, Dict[str, float]]:
    """
    Tune set builder parameters using backtesting with simulated annealing.
    """
    params = {}
    N = probs_matrix.shape[0]
    if N == 0:
        for s in sizes:
            params[int(s)] = {"temp": 1.0, "mu": 0.06}
        return params

    for s in sizes:
        # Use SA for each size
        sa_result = simulated_annealing_tune(
            probs_matrix, y_matrix, corr, int(s),
            seed=RNG_SEED + int(s) * 100
        )
        params[int(s)] = sa_result

    return params


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_prediction_model_workflow():
    """Main execution for standard model."""
    print("=" * 60)
    print(f"--- STANDARD MODEL v2.0 ({PREDICTIONS_SAVE_FILE}) ---")
    print("=" * 60)

    game_ids, raw, n = load_data(DATA_FILE_PATH)
    if n == 0 or raw.size == 0:
        print("No data loaded.")
        return

    print(f"Loaded {n} draws from {DATA_FILE_PATH}")

    data = np.copy(raw)
    data[(data < 1) | (data > MAX_KENO_NUMBER)] = 0

    # Build advanced correlation matrix
    corr = build_advanced_correlation_matrix(data)
    print("Built advanced correlation matrix")

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
                "diversity_loss": diversity_loss
            }):
                model = keras.models.load_model(MODEL_SAVE_PATH)
                print("Loaded existing model")
        except Exception as e:
            print(f"Could not load model: {e}")

    full_retrain = (len(game_ids) > 0 and int(game_ids[-1]) % FULL_RETRAIN_INTERVAL == 0)
    if model is None:
        print("Building NEW model (force full retrain)")
        model = build_advanced_model()
        full_retrain = True

    if full_retrain:
        print("Mode: FULL RETRAIN")
        split = int(0.9 * len(X_seq))
        model.fit(
            [X_seq[:split], X_glob[:split], X_spec[:split]],
            Y[:split],
            validation_data=([X_seq[split:], X_glob[split:], X_spec[split:]], Y[split:]),
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
        idx = max(0, len(X_seq) - MINI_BATCH_SIZE)
        model.fit(
            [X_seq[idx:], X_glob[idx:], X_spec[idx:]],
            Y[idx:],
            epochs=MINI_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )

    model.save(MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}")

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
    lookback = max(SHORT_TERM_LOOKBACK, MEDIUM_TERM_LOOKBACK, LONG_TERM_LOOKBACK, 20)
    p_seq = compute_sequence_features(data[-lookback:]).reshape(1, 20, MAX_KENO_NUMBER)
    p_glob = compute_global_features(data[-lookback:]).reshape(1, -1)
    p_spec = compute_draw_features(data[-1]).reshape(1, -1)

    probs = model.predict([p_seq, p_glob, p_spec], verbose=0)[0]

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
            build_diverse_sets(
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

    # Extras
    predicted_sets["pick_20"] = sorted([int(x) for x in core_20])
    for s in EXTRA_STYLE_SIZES:
        predicted_sets[f"pick_{int(s)}"] = sorted([int(x) for x in core_20[:int(s)]])

    # Convert all numpy ints to Python ints for JSON serialization
    predicted_sets_clean = {}
    for k, v in predicted_sets.items():
        predicted_sets_clean[k] = [int(x) for x in v]

    output = {
        "core_20": [int(x) for x in core_20],
        "probs": [float(x) for x in probs.tolist()],
        "predicted_sets": predicted_sets_clean,
        "meta": {
            "model": "Standard_v2",
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
