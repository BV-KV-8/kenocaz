"""
Display scores and predictions in terminal
"""
import json
import os

SCORES_CSV = "scores.csv"
GAMES_CSV = "games.csv"
PRED_FILE_STD = "pred_standard.json"
PRED_FILE_SMART = "pred_smart.json"
PRED_FILE_ENSEMBLE = "pred_ensemble.json"
SCORE_STYLES = ("pick_3", "pick_5", "pick_8")


def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}


def show_latest_results():
    """Show the most recent game results and scores"""
    print("\n" + "=" * 60)
    print("LATEST RESULTS")
    print("=" * 60)
    
    # Load predictions
    std_data = load_json(PRED_FILE_STD)
    smart_data = load_json(PRED_FILE_SMART)
    ensemble_data = load_json(PRED_FILE_ENSEMBLE)
    
    # Show next predictions
    print("\nNEXT PREDICTIONS:")
    
    def show_model_preds(name, data):
        if not isinstance(data, dict) or "predicted_sets" not in data:
            return
        ps = data["predicted_sets"]
        print(f"\n  {name}:")
        for style in SCORE_STYLES:
            if style in ps and isinstance(ps[style], list):
                nums_str = ", ".join(str(n) for n in ps[style])
                print(f"    {style}: {nums_str}")
    
    show_model_preds("Standard", std_data)
    show_model_preds("Smart", smart_data)
    show_model_preds("Ensemble", ensemble_data)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    show_latest_results()
