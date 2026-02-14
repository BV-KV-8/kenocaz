Keno Prediction System

Complete ML-based prediction system for Casino Arizona Keno (McKellips location).
Files Overview
Core System

    scraper.py - Fetches game data from kenousa.com, stores in games.csv

    mllogic_standard.py - Standard LSTM prediction model

    mllogic_smart.py - Advanced model with correlation analysis

    scoring.py - Tracks predictions and sends Telegram updates

    monitor.py - Watches games.csv and triggers ML workflow

    launcher.py - Starts entire system (scraper + monitor)

Utilities

    convert_csv.py - Converts old CSV format to new format

CSV Format

New Format (games.csv):

text
draw_id,timestamp,n1,n2,n3,...,n20
draw_381,12/22/25 21:01:35,10,16,19,...,71

Each number has its own column (n1 through n20) for easy analysis.
Quick Start
1. Initial Setup

bash
# If you have old format CSV, convert it:
python convert_csv.py games_old.csv games.csv

# Or let scraper fetch fresh data:
python scraper.py

2. Run Complete System

bash
python launcher.py

This starts:

    Scraper (monitoring for new games)

    Monitor (running ML models on updates)

3. Run Components Separately

bash
# Just scraper
python scraper.py

# Just ML predictions
python mllogic_standard.py
python mllogic_smart.py

# Just scoring
python scoring.py

# Just monitor
python monitor.py

Configuration
Telegram Notifications

Edit scoring.py:

python
TELEGRAM_BOT_TOKEN = "your_token"
TELEGRAM_CHAT_ID = "your_chat_id"

Model Training

Edit model files for training parameters:

    FULL_RETRAIN_INTERVAL - Retrain every N games

    FULL_EPOCHS - Training epochs for full retrain

    MINI_EPOCHS - Training epochs for incremental update

Output Files

    games.csv - Historical game data

    pred_standard.json - Standard model predictions

    pred_smart.json - Smart model predictions

    active_predictions.json - Predictions being tracked

    scores.csv - Historical prediction performance

    model_standard.keras - Trained standard model

    model_smart.keras - Trained smart model

Workflow

    Scraper fetches new game → Updates games.csv

    Monitor detects change → Triggers ML workflow

    Both models train/update → Generate predictions

    Scoring evaluates past predictions → Logs results

    Telegram notification sent with results + new predictions

Requirements

bash
pip install tensorflow pandas numpy requests watchdog

Troubleshooting

"No module named tensorflow"

bash
pip install tensorflow

"Old CSV format detected"

bash
python convert_csv.py

Monitor not triggering

    Check that games.csv exists

    Ensure watchdog is installed: pip install watchdog

Notes

    System tracks predictions for 5 games (TRACKING_DURATION)

    Full retraining happens every 12 games by default

    Predictions use pick-10 style for scoring

    Draw IDs wrap at 999 (returns to 1)
