import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WORKFLOW = [
    ("Standard Model", ["python", "mllogic_standard.py"]),
    ("Smart Model", ["python", "mllogic_smart.py"]),
    ("Scoring", ["python", "scoring.py"]),
    ("Optimizer", ["python", "optimizer.py"]),
    ("Show Results", ["python", "show_results.py"]),  # Added
]


class GameDataHandler(FileSystemEventHandler):
    """Monitor games.csv for changes and trigger workflow"""

    def __init__(self):
        self.last_modified = time.time()
        self.cooldown = 5

    def on_modified(self, event):
        if not event.src_path.endswith("games.csv"):
            return

        now = time.time()
        if now - self.last_modified < self.cooldown:
            return
        self.last_modified = now

        print(f"\n[{time.strftime('%H:%M:%S')}] Data Update Detected! Running Workflow...\n")
        run_workflow()


def run_workflow():
    try:
        total = len(WORKFLOW)
        for idx, (name, cmd) in enumerate(WORKFLOW, start=1):
            print(f" >> [{idx}/{total}] Running {name}...")
            subprocess.run(cmd, check=True)
        print("\n >> Workflow Complete!\n")
    except subprocess.CalledProcessError as e:
        print(f" [!] Workflow error: {e}\n")
    except Exception as e:
        print(f" [!] Unexpected error: {e}\n")


def run_monitor():
    print("=" * 60)
    print("--- Keno Monitor Started (Unified Repo) ---")
    print("=" * 60)
    print("Watching games.csv...\n")

    print("[Initial Run] Executing workflow...\n")
    run_workflow()
    print("[Initial Run] Complete\n")

    handler = GameDataHandler()
    observer = Observer()
    observer.schedule(handler, path=".", recursive=False)
    observer.start()

    print("Monitor active. Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping monitor...")
        observer.stop()
        observer.join()


if __name__ == "__main__":
    if not os.path.exists("games.csv"):
        print("Error: games.csv not found!")
        print("Run scraper.py first to fetch data.")
        raise SystemExit(1)

    run_monitor()
