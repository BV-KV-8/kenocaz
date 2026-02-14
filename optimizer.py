import json
import os
import csv
import numpy as np

SCORES_FILE = "scores.csv"
STATE_FILE = "brainstate.json"
GAMES_FILE = "games.csv"


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                s = json.load(f)
            gb = s.get("global_bias", None)
            if isinstance(gb, list) and len(gb) >= 81:
                s["global_bias"] = [float(x) for x in gb[:81]]
                return s
        except:
            pass
    return {"global_bias": [1.0] * 81, "meta": {"source": "init"}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)


def read_recent_hit_numbers(limit_rows=50):
    hit_nums = []
    if not os.path.exists(SCORES_FILE):
        return hit_nums

    try:
        with open(SCORES_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)[-limit_rows:]
        for r in rows:
            s = r.get("hit_numbers", "")
            if not s:
                continue
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    hit_nums.extend([int(x) for x in arr])
            except:
                pass
    except:
        pass

    return hit_nums


def fallback_recent_draw_numbers(limit_draws=40):
    nums = []
    if not os.path.exists(GAMES_FILE):
        return nums
    try:
        with open(GAMES_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)[-limit_draws:]
        for r in rows:
            for i in range(1, 21):
                k = f"n{i}"
                v = r.get(k, "")
                if v is None or str(v).strip() == "":
                    continue
                nums.append(int(str(v).strip()))
    except:
        pass
    return nums


def analyze_performance():
    state = load_state()
    gb = state.get("global_bias", [1.0] * 81)
    if not (isinstance(gb, list) and len(gb) >= 81):
        gb = [1.0] * 81

    hit_nums = read_recent_hit_numbers(limit_rows=60)
    if not hit_nums:
        hit_nums = fallback_recent_draw_numbers(limit_draws=60)

    if not hit_nums:
        state["meta"] = {"source": "no_data"}
        state["global_bias"] = gb[:81]
        save_state(state)
        print(" [Brain] No data. State unchanged.")
        return state

    avg = float(np.mean(hit_nums))

    # Adjust gently each run
    def boost(lo, hi, up, down):
        for i in range(lo, hi + 1):
            gb[i] = min(1.5, gb[i] * up)
        for i in range(1, 81):
            if i < lo or i > hi:
                gb[i] = max(0.5, gb[i] * down)

    if avg > 45:
        print(" [Brain] Trend: high numbers.")
        boost(41, 80, up=1.03, down=0.99)
    elif avg < 35:
        print(" [Brain] Trend: low numbers.")
        boost(1, 40, up=1.03, down=0.99)
    else:
        # decay toward 1.0
        for i in range(1, 81):
            gb[i] = gb[i] * 0.98 + 0.02

    state["global_bias"] = [float(x) for x in gb[:81]]
    state["meta"] = {"source": "scores_or_games", "avg_hit_number": avg}
    save_state(state)

    print(" [Brain] Updated brainstate.json")
    return state


if __name__ == "__main__":
    analyze_performance()
