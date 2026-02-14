import json
import csv
import os
import requests
import statistics
from collections import deque
import numpy as np

# --- CONFIGURATION ---
PRED_FILE_STD = "pred_standard.json"
PRED_FILE_SMART = "pred_smart.json"
PRED_FILE_ENSEMBLE = "pred_ensemble.json"
ACTIVE_PREDICTIONS_FILE = "active_predictions.json"
SCORES_CSV = "scores.csv"
GAMES_CSV = "games.csv"
TRACKING_DURATION = 5

# Expected hits for random selection:
EXPECTED_VALS = {3: 0.75, 5: 1.25, 8: 2.0, 10: 2.5, 20: 5.0}
SCORE_STYLES = ("pick_3", "pick_5", "pick_8")
ALSO_SCORE = ("pick_10",)
CORR_LOOKBACK = 250

# --- TELEGRAM ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if (not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID) and os.path.exists("telegram.json"):
    try:
        with open("telegram.json", "r") as f:
            tcfg = json.load(f)
        TELEGRAM_BOT_TOKEN = str(tcfg.get("bot_token", "")).strip()
        TELEGRAM_CHAT_ID = str(tcfg.get("chat_id", "")).strip()
    except:
        pass

TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage" if TELEGRAM_BOT_TOKEN else f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN[:20]}/sendMessage"


def escape_markdown(text: str) -> str:
    for char in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}']:
        text = text.replace(char, "\\")
        return text


def send_telegram(msg: str) -> None:
    """Send message to Telegram. Called on EVERY game."""
    print(f"[DEBUG] Telegram bot configured: {bool(TELEGRAM_BOT_TOKEN)}")
    print(f"[DEBUG] Telegram chat_id configured: {bool(TELEGRAM_CHAT_ID)}")
    print(f"[DEBUG] Telegram URL set: {bool(TELEGRAM_URL)}")
    print(f"[DEBUG] Message length: {len(msg)}")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or not TELEGRAM_URL:
        print("[!] Telegram disabled: missing credentials")
        print("[DEBUG] Skipping Telegram send")
        return

    try:
        print("[DEBUG] Sending POST request to Telegram API...")
        print(f"[DEBUG] URL: {TELEGRAM_URL[:80]}...")
        resp = requests.post(
            TELEGRAM_URL,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )

        print(f"[DEBUG] Response status code: {resp.status_code}")
        print(f"[DEBUG] Response text (first 200 chars): {resp.text[:200] if resp.text else 'No response'}")

        if resp.status_code != 200:
            print(f"[!] Telegram HTTP {resp.status_code}: {resp.text[:300]}")
        else:
            print("[OK] Telegram message sent successfully.")
    except Exception as e:
        print(f"[!] Telegram failed: {e}")


def load_games(tail_games: int = 800):
    games = []
    if not os.path.exists(GAMES_CSV):
        return games

    try:
        last_rows = deque(maxlen=tail_games)
        with open(GAMES_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                last_rows.append(r)

        for r in last_rows:
            did_raw = r.get("draw_id", "")
            ts_raw = r.get("timestamp", "N/A")
            if did_raw is None or str(did_raw).strip() == "":
                continue

            draw_id = int(str(did_raw).strip())
            timestamp = str(ts_raw).strip() if ts_raw is not None else "N/A"

            nums = []
            ok = True
            for i in range(1, 21):
                v = r.get(f"n{i}", None)
                if v is None or str(v).strip() == "":
                    ok = False
                    break
                nums.append(int(str(v).strip()))

            if not ok or len(nums) != 20:
                continue

            games.append({"id": draw_id, "timestamp": timestamp, "numbers": nums})
    except Exception as e:
        print(f"[!] Error loading {GAMES_CSV}: {e}")

    return games


def build_corr_from_games(games, lookback=250):
    m = np.zeros((81, 81), dtype=float)
    recent = games[-lookback:] if len(games) >= lookback else games

    for g in recent:
        draw = g["numbers"]
        for i in draw:
            if not (1 <= i <= 80):
                continue
            for j in draw:
                if not (1 <= j <= 80) or i == j:
                    continue
                m[i, j] += 1.0

    mx = float(np.max(m))
    if mx > 0:
        m /= mx

    return m


def get_next_game_id(current_id: int) -> int:
    return 1 if current_id == 999 else current_id + 1


def calc_game_age(start_id: int, current_id: int) -> int:
    if current_id >= start_id:
        return current_id - start_id + 1
    return (999 - start_id) + current_id + 1


def ensure_scores_header():
    if os.path.exists(SCORES_CSV) and os.path.getsize(SCORES_CSV) > 0:
        return

    with open(SCORES_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "model", "style", "pick_size", "hits", "expected", "diff", "hit_numbers"])


def record_score(timestamp: str, model: str, style: str, pick_size: int, hits: int, hit_numbers):
    ensure_scores_header()
    expected = float(EXPECTED_VALS.get(int(pick_size), pick_size / 4.0))
    diff = float(hits) - expected

    try:
        with open(SCORES_CSV, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([timestamp, model, style, pick_size, hits, expected, diff, json.dumps(hit_numbers)])
        print(f"[+] Logged: {model} {style} ({hits} hits, Diff {diff:+.2f})")
    except Exception as e:
        print(f"[!] CSV Write Error: {e}")


def rolling_mean_diff(model: str, style: str, limit: int = 200) -> float:
    if not os.path.exists(SCORES_CSV):
        return 0.0

    diffs = []
    try:
        with open(SCORES_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model") == model and row.get("style") == style:
                    try:
                        diffs.append(float(row.get("diff", "0")))
                    except:
                        pass
    except:
        return 0.0

    recent = diffs[-limit:]
    return float(statistics.mean(recent)) if recent else 0.0


def topk_core(probs, k=20):
    p = np.asarray(probs, dtype=float)
    idx = np.argsort(p)[::-1][:k]
    return [int(i + 1) for i in idx]


def build_sets_from_core20(core20, probs, corr, sizes=(3, 5, 8), num_candidates=30, temp=1.0, mu=0.06, seed=999):
    rng = np.random.default_rng(seed)
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 1e-9, 1.0)

    core = [int(x) for x in core20 if 1 <= int(x) <= 80]
    core = list(dict.fromkeys(core))

    def weight_for(cand, chosen):
        w = (p[cand - 1]) ** (1.0 / max(1e-6, float(temp)))
        if chosen:
            pen = 0.0
            for ex in chosen:
                pen += float(corr[ex, cand])
            w *= float(np.exp(-mu * pen))
        return max(1e-12, float(w))

    def make_candidate(k):
        chosen = []
        available = core[:]
        while len(chosen) < k and available:
            weights = np.array([weight_for(c, chosen) for c in available], dtype=float)
            s = float(weights.sum())
            if s <= 0:
                pick = int(available[0])
            else:
                pick = int(rng.choice(available, p=(weights / s)))
            chosen.append(pick)
            available.remove(pick)
        return sorted(chosen)

    def set_score(nums):
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
        for _ in range(int(num_candidates)):
            cand = make_candidate(int(k))
            sc = set_score(cand)
            if sc > best_sc:
                best_sc = sc
                best = cand
        out[f"pick_{int(k)}"] = best if best is not None else sorted(core[: int(k)])

    return out


def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}


def write_json(path, obj):
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=4)
    except Exception as e:
        print(f"[!] Could not write {path}: {e}")


def make_ensemble(std_data, smart_data, corr):
    ps = std_data.get("probs", None)
    pm = smart_data.get("probs", None)

    if not (isinstance(ps, list) and len(ps) == 80 and isinstance(pm, list) and len(pm) == 80):
        return {}

    w_std = 1.0 + max(-0.5, min(0.5, rolling_mean_diff("Standard", "pick_8", limit=200)))
    w_smt = 1.0 + max(-0.5, min(0.5, rolling_mean_diff("Smart", "pick_8", limit=200)))

    denom = float(w_std + w_smt) if (w_std + w_smt) != 0 else 1.0
    w_std /= denom
    w_smt /= denom

    probs = (w_std * np.asarray(ps, dtype=float)) + (w_smt * np.asarray(pm, dtype=float))
    core20 = topk_core(probs, 20)

    sets = build_sets_from_core20(core20, probs, corr, sizes=(3, 5, 8), num_candidates=40, temp=1.0, mu=0.06, seed=777)
    sets["pick_20"] = sorted([int(x) for x in core20])
    sets["pick_10"] = sorted([int(x) for x in core20[:10]])

    return {
        "core_20": [int(x) for x in core20],
        "probs": [float(x) for x in probs.tolist()],
        "predicted_sets": sets,
        "meta": {"model": "Ensemble", "weights": {"standard": w_std, "smart": w_smt}},
    }


def run_scoring_workflow():
    print("--- SCORING (TSR) ---")

    games = load_games()
    if not games:
        print("No games data.")
        return

    latest_game = games[-1]
    latest_id = int(latest_game["id"])
    target_id = get_next_game_id(latest_id)
    actual = set(latest_game["numbers"])

    corr = build_corr_from_games(games, lookback=CORR_LOOKBACK)

    std_data = load_json(PRED_FILE_STD)
    smart_data = load_json(PRED_FILE_SMART)
    ensemble_data = make_ensemble(std_data, smart_data, corr)
    if ensemble_data:
        write_json(PRED_FILE_ENSEMBLE, ensemble_data)

    active_preds = []
    if os.path.exists(ACTIVE_PREDICTIONS_FILE):
        try:
            with open(ACTIVE_PREDICTIONS_FILE, "r") as f:
                active_preds = json.load(f)
        except:
            active_preds = []

    def register(model_name: str, data: dict):
        if not isinstance(data, dict) or "predicted_sets" not in data:
            return
        if any(p.get("start_game_id") == target_id and p.get("model") == model_name for p in active_preds):
            return
        active_preds.append({"start_game_id": target_id, "model": model_name, "picks": data["predicted_sets"], "results": []})

    register("Standard", std_data)
    register("Smart", smart_data)
    register("Ensemble", ensemble_data)

    for pred in active_preds:
        try:
            start_id = int(pred.get("start_game_id"))
        except:
            continue

        age = calc_game_age(start_id, latest_id)
        if not (0 < age <= TRACKING_DURATION):
            continue

        pred.setdefault("results", [])
        picks = pred.get("picks", {})
        if not isinstance(picks, dict):
            continue

        styles = [s for s in SCORE_STYLES if s in picks] + [s for s in ALSO_SCORE if s in picks]
        for style in styles:
            if any(isinstance(r, dict) and r.get("game_id") == latest_id and r.get("style") == style for r in pred["results"]):
                continue

            try:
                nums = [int(x) for x in picks.get(style, [])]
            except:
                nums = []

            picked = set(nums)
            hit_numbers = sorted(list(picked.intersection(actual)))
            hits = len(hit_numbers)

            try:
                pick_size = int(str(style).replace("pick_", ""))
            except:
                pick_size = len(nums)

            record_score(latest_game["timestamp"], str(pred.get("model", "Unknown")), style, pick_size, hits, hit_numbers)
            pred["results"].append({"game_id": latest_id, "style": style, "hits": hits, "pick_size": pick_size, "hit_numbers": hit_numbers})

    active_preds = [
        p
        for p in active_preds
        if isinstance(p, dict)
        and "start_game_id" in p
        and calc_game_age(int(p["start_game_id"]), latest_id) <= TRACKING_DURATION
    ]

    try:
        with open(ACTIVE_PREDICTIONS_FILE, "w") as f:
            json.dump(active_preds, f, indent=4)
    except Exception as e:
        print(f"[!] Error writing {ACTIVE_PREDICTIONS_FILE}: {e}")

    # ========================================
    # TERMINAL OUTPUT: Show scores and predictions
    # ========================================
    print("")
    print("=" * 60)
    print(f"RESULT: #{latest_id} at {latest_game['timestamp']}")
    print("-" * 60)

    drawn_nums = sorted(latest_game['numbers'])
    drawn_str = ", ".join(str(n) for n in drawn_nums)
    print(f"  Drawn: {drawn_str}")

    print("")
    print("SCORES:")
    for model_name in ("Standard", "Smart", "Ensemble"):
        p = next((x for x in active_preds if x.get("model") == model_name), None)
        if not p:
            continue

        for style in SCORE_STYLES:
            r = next((x for x in p.get("results", []) if x.get("game_id") == latest_id and x.get("style") == style), None)
            if r:
                hits = r.get('hits', 0)
                pick_size = r.get('pick_size', 0)
                hit_nums = r.get('hit_numbers', [])
                print(f"  {model_name} {style}: {hits}/{pick_size} hits -> {hit_nums}")

    print("")
    print("NEXT PREDICTIONS:")

    def show_model_preds(name, data):
        if not isinstance(data, dict) or "predicted_sets" not in data:
            return
        ps = data["predicted_sets"]
        print(f"  {name}:")
        for style in SCORE_STYLES:
            if style in ps and isinstance(ps[style], list):
                nums_str = ", ".join(str(n) for n in ps[style])
                print(f"    {style}: {nums_str}")

    show_model_preds("Standard", std_data)
    show_model_preds("Smart", smart_data)
    show_model_preds("Ensemble", ensemble_data)

    print("=" * 60)
    print("")

    # ========================================
    # TELEGRAM: ALWAYS send on every game
    # ========================================
    msg = []
    msg.append(f"📢 *TSR RESULT: #{latest_id}*")
    msg.append(f"{escape_markdown(latest_game['timestamp'])}")

    drawn_str = ", ".join(str(n) for n in sorted(latest_game['numbers']))
    msg.append(f"Drawn: {drawn_str}")

    lines = []
    for model_name in ("Standard", "Smart", "Ensemble"):
        p = next((x for x in active_preds if x.get("model") == model_name), None)
        if not p:
            continue

        for style in SCORE_STYLES:
            r = next((x for x in p.get("results", []) if x.get("game_id") == latest_id and x.get("style") == style), None)
            if r:
                hit_str = ", ".join(str(n) for n in r.get('hit_numbers', []))
                lines.append(f"✅ {model_name} {style}: {r.get('hits', 0)}/{r.get('pick_size', 0)} -> {hit_str}")

    if lines:
        msg.append("")
        msg.extend(lines)
    else:
        msg.append("")
        msg.append("_(No active forecasts scored)_")

    msg.append("")
    msg.append(f"🔮 *NEXT PREDICTIONS: #{target_id}*")

    def add_next(model_name, data):
        if not isinstance(data, dict) or "predicted_sets" not in data:
            return
        ps = data["predicted_sets"]
        parts = []
        for style in SCORE_STYLES:
            if style in ps and isinstance(ps[style], list):
                nums_str = ", ".join(str(n) for n in ps[style])
                parts.append(f"{style}: {nums_str}")
        if parts:
            msg.append(f"*{model_name}:*")
            for part in parts:
                msg.append(f"  {part}")

    add_next("Standard", std_data)
    add_next("Smart", smart_data)
    add_next("Ensemble", ensemble_data)

    send_telegram("\n".join(msg))
    print("--- SCORING COMPLETE ---")


if __name__ == "__main__":
    run_scoring_workflow()
