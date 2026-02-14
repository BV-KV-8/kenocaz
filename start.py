import subprocess
import sys
import time
import os

def main():
    print("==================================================")
    print("   KENO VISION AUTO-LAUNCHER")
    print("==================================================")

    # 1. Check for required files
    required_files = ["scraper.py", "monitor.py", "mllogic_standard.py", "mllogic_smart.py", "scoring.py"]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"Error: Missing files: {missing}")
        return

    processes = []

    try:
        # 2. Launch Scraper
        print("  [>] Launching Scraper (Background)...")
        p_scraper = subprocess.Popen([sys.executable, "scraper.py"])
        processes.append(p_scraper)

        # Short pause to let scraper create CSV if needed
        time.sleep(2)

        # 3. Launch Monitor
        print("  [>] Launching Monitor (Background)...")
        p_monitor = subprocess.Popen([sys.executable, "monitor.py"])
        processes.append(p_monitor)

        print("\nAll systems GO. Logs will appear below.")
        print("Press Ctrl+C to shut down.")
        print("--------------------------------------------------")

        # 4. Keep alive loop & Auto-Restart logic
        while True:
            time.sleep(1)

            if p_scraper.poll() is not None:
                print("\n[!] Scraper stopped unexpectedly. Restarting...")
                p_scraper = subprocess.Popen([sys.executable, "scraper.py"])
                processes[0] = p_scraper

            if p_monitor.poll() is not None:
                print("\n[!] Monitor stopped unexpectedly. Restarting...")
                p_monitor = subprocess.Popen([sys.executable, "monitor.py"])
                processes[1] = p_monitor

    except KeyboardInterrupt:
        print("\n\nShutting down Keno Vision...")
        for p in processes:
            p.terminate()
        print("Goodbye.")

if __name__ == "__main__":
    main()
