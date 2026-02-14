#!/usr/bin/env python3
"""
Main Launcher for Keno Prediction System
Starts scraper and monitor in parallel
"""
import subprocess
import time
import sys
from threading import Thread

def run_scraper():
    """Run scraper in subprocess"""
    print("[>] Launching Scraper (Background)...")
    try:
        subprocess.run(['python', 'scraper.py'])
    except KeyboardInterrupt:
        pass

def run_monitor():
    """Run monitor in subprocess"""
    time.sleep(2)  # Let scraper start first
    print("  [>] Launching Monitor (Background)...")
    try:
        subprocess.run(['python', 'monitor.py'])
    except KeyboardInterrupt:
        pass

def main():
    print("="*60)
    print("KENO PREDICTION SYSTEM - LAUNCHER")
    print("="*60)
    print()

    # Check if games.csv exists
    import os
    if not os.path.exists('games.csv'):
        print("games.csv not found. Creating from scraper...")

    print("All systems GO. Logs will appear below.")
    print("Press Ctrl+C to shut down.")
    print("-"*50)
    print()

    # Start both processes
    scraper_thread = Thread(target=run_scraper, daemon=True)
    monitor_thread = Thread(target=run_monitor, daemon=True)

    scraper_thread.start()
    monitor_thread.start()

    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n[Shutdown] Stopping all processes...")
        sys.exit(0)

if __name__ == "__main__":
    main()
