import requests
import time
import csv
import os
from datetime import datetime


class KenoScraper:
    """
    Casino Arizona Keno Scraper

    CANONICAL OUTPUT FORMAT:
      draw_id,timestamp,n1,n2,...,n20

    Game IDs cycle 1-999 and repeat. Dedupe by (draw_id, timestamp).
    """

    API_URL = "https://kenousa.com/php/getDraws.php"
    LAST_GAME_URL = "https://kenousa.com/php/getDraw.php"

    CSV_PATH = "games.csv"
    CASINO_ID = "CasinoArizona"
    GAME_NAME = "McKellips"

    STARTUP_SYNC_GAMES = 1000
    MONITOR_POLL_SECONDS = 10
    BATCH_SIZE = 25
    REQUEST_TIMEOUT = 15

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": f"https://kenousa.com/games/{self.CASINO_ID}/{self.GAME_NAME}/draws.php",
            }
        )

        self._ensure_csv()
        # Key by (draw_id, timestamp) so we don't miss draws when draw_id repeats
        self.existing_keys = self._load_existing_keys()
        print(f"[Init] Loaded {len(self.existing_keys)} existing games from CSV.")

    def _ensure_csv(self):
        if not os.path.exists(self.CSV_PATH):
            with open(self.CSV_PATH, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["draw_id", "timestamp"] + [f"n{i}" for i in range(1, 21)])

    def _load_existing_keys(self):
        keys = set()
        if not os.path.exists(self.CSV_PATH):
            return keys

        try:
            with open(self.CSV_PATH, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    ts = r.get("timestamp")
                    if ts is None:
                        continue

                    ts_s = str(ts).strip()
                    if ts_s:
                        keys.add(ts_s)
        except:
            pass

        return keys

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse MM/DD/YY HH:MM:SS format to datetime for sorting."""
        ts_str = str(ts_str).strip()
        try:
            return datetime.strptime(ts_str, "%m/%d/%y %H:%M:%S")
        except:
            return datetime(1970, 1, 1)

    def _extract_timestamp(self, game: dict) -> str:
        # Always return a string (never int/None)
        for k in ("DrawTime", "DrawDateTime", "DrawDate", "Time", "DateTime"):
            v = game.get(k)
            if v is not None:
                s = str(v).strip()
                if s:
                    return s
        return "N/A"

    def get_latest_game_ordinal(self) -> int:
        try:
            payload = {"Casino": self.CASINO_ID, "GameName": self.GAME_NAME, "Ordinal": -1}
            r = self.session.post(self.LAST_GAME_URL, data=payload, timeout=self.REQUEST_TIMEOUT)
            data = r.json()
            return int(data.get("OrdinalNumber", 0) or 0)
        except:
            return 0

    def fetch_history(self, start_ordinal: int, num_games: int):
        """
        Fetch EXACTLY up to num_games (1000 on startup), walking backward by ordinal windows.
        DEDUPES within the fetched batch to prevent API overlap from creating duplicates.
        """
        collected = []
        seen_in_fetch = set()  # Track (draw_id, timestamp) within THIS fetch to prevent dupes
        current = start_ordinal if start_ordinal > 0 else self.get_latest_game_ordinal()

        print(f" Fetching {num_games} games starting from ordinal {current}...")

        while len(collected) < num_games and current > 0:
            payload = {
                "Casino": self.CASINO_ID,
                "GameName": self.GAME_NAME,
                "OrdinalStart": current,
                "OrdinalEnd": max(1, current - (self.BATCH_SIZE - 1)),
                "Direction": "Back",
            }

            try:
                r = self.session.post(self.API_URL, data=payload, timeout=self.REQUEST_TIMEOUT)
                data = r.json()
                items = data if isinstance(data, list) else list(data.values()) if isinstance(data, dict) else []
                if not items:
                    break

                for game in items:
                    if not isinstance(game, dict) or "Draw" not in game:
                        continue

                    draw_id = int(game.get("GameNumber", 0) or 0)
                    if draw_id <= 0:
                        continue

                    ts = self._extract_timestamp(game)
                    key = ts  # Use timestamp only for dedupe (draw_ids wrap 999->1)

                    # Skip if already seen in this fetch (API overlap)
                    if key in seen_in_fetch:
                        continue

                    raw = game.get("Draw", [])[:20]
                    try:
                        nums = [int(x) for x in raw]
                    except:
                        continue

                    if len(nums) != 20:
                        continue

                    collected.append({"draw_id": draw_id, "timestamp": ts, "numbers": nums})
                    seen_in_fetch.add(key)

                    if len(collected) >= num_games:
                        break

                current -= self.BATCH_SIZE
                time.sleep(0.35)

            except Exception as e:
                print(f" [!] Fetch error: {e}")
                break

        return collected

    def append_to_csv(self, games):
        """
        Append missing games to CSV in proper chronological order.
        Only write games that aren't already in existing_keys (from CSV).
        Sort by parsed datetime to ensure proper chronological ordering.
        """
        if not games:
            return 0

        added = 0
        skipped = 0

        # Filter out duplicates and sort by actual datetime
        new_games = []
        for g in games:
            did = int(g["draw_id"])
            ts = str(g["timestamp"]).strip()
            key = ts  # Use timestamp only for dedupe (draw_ids wrap 999->1)

            if key in self.existing_keys:
                skipped += 1
                continue

            new_games.append(g)
            self.existing_keys.add(key)

        # Sort by parsed datetime for proper chronological order
        new_games.sort(key=lambda g: self._parse_timestamp(g["timestamp"]))

        # Write to CSV
        with open(self.CSV_PATH, "a", newline="") as f:
            w = csv.writer(f)
            for g in new_games:
                row = [int(g["draw_id"]), str(g["timestamp"]).strip()] + [int(x) for x in g["numbers"]]
                w.writerow(row)
                added += 1

        if added > 0:
            print(f" [+] Added {added} new games.")
        if skipped > 0:
            print(f" [=] Skipped {skipped} duplicates already in CSV.")

        return added

    def run(self):
        print(f"--- Scraper ({self.CASINO_ID}/{self.GAME_NAME}) ---")

        print(f"\n[Startup] Fetching last {self.STARTUP_SYNC_GAMES} games...")
        history = self.fetch_history(0, self.STARTUP_SYNC_GAMES)
        print(f"[Startup] Fetched {len(history)}/{self.STARTUP_SYNC_GAMES} unique games from API.")
        added = self.append_to_csv(history)
        print(f"[Startup] Complete. Added {added} new games to CSV.")

        print("\n[Monitor] Listening for new games...")
        last = self.get_latest_game_ordinal()

        while True:
            try:
                curr = self.get_latest_game_ordinal()
                if curr > last:
                    print(f"\n[NEW GAME] Detected! Global ordinal: {curr}")
                    recent = self.fetch_history(curr, 30)
                    self.append_to_csv(recent)
                    last = curr

                time.sleep(self.MONITOR_POLL_SECONDS)

            except KeyboardInterrupt:
                print("\n[Shutdown] Stopping scraper...")
                break
            except Exception as e:
                print(f" [!] Monitor error: {e}")
                time.sleep(self.MONITOR_POLL_SECONDS)


if __name__ == "__main__":
    KenoScraper().run()
