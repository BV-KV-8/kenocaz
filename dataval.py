import csv
import json
import os
import logging
from collections import Counter

# Configuration
CSV_FILE = 'games.csv'
LOG_FILE = 'data_validation.log'

# Setup Logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def validate_data():
    print(f"--- Running Data Validation on {CSV_FILE} ---")
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    draws = []
    ids = []
    errors = 0
    warnings = 0

    try:
        with open(CSV_FILE, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        print(f"Scanning {len(rows)} records...")

        for i, row in enumerate(rows):
            line_num = i + 1

            # Check 1: Column Count
            if len(row) < 3:
                msg = f"Line {line_num}: Invalid format (columns < 3)"
                logging.error(msg)
                print(f"[!] {msg}")
                errors += 1
                continue

            # Check 2: ID Format
            draw_id_str = row[0]
            try:
                draw_id = int(draw_id_str.replace('draw_', ''))
                ids.append(draw_id)
            except ValueError:
                msg = f"Line {line_num}: Invalid Draw ID format '{draw_id_str}'"
                logging.error(msg)
                print(f"[!] {msg}")
                errors += 1

            # Check 3: JSON Numbers
            try:
                numbers = json.loads(row[2])
                if not isinstance(numbers, list):
                    raise ValueError("Not a list")
            except Exception as e:
                msg = f"Line {line_num}: Invalid JSON numbers - {e}"
                logging.error(msg)
                print(f"[!] {msg}")
                errors += 1
                continue

            # Check 4: Number Count (Must be 20)
            if len(numbers) != 20:
                msg = f"Line {line_num}: Draw {draw_id} has {len(numbers)} numbers (Expected 20)"
                logging.error(msg)
                print(f"[!] {msg}")
                errors += 1

            # Check 5: Range (1-80)
            out_of_range = [n for n in numbers if n < 1 or n > 80]
            if out_of_range:
                msg = f"Line {line_num}: Draw {draw_id} contains invalid numbers {out_of_range}"
                logging.error(msg)
                print(f"[!] {msg}")
                errors += 1

            # Check 6: Duplicates within draw
            if len(numbers) != len(set(numbers)):
                msg = f"Line {line_num}: Draw {draw_id} contains duplicate numbers"
                logging.error(msg)
                print(f"[!] {msg}")
                errors += 1

        # Check 7: Duplicate Draw IDs globally
        id_counts = Counter(ids)
        dupe_ids = [id for id, count in id_counts.items() if count > 1]
        if dupe_ids:
            msg = f"Found {len(dupe_ids)} duplicate Draw IDs (e.g., {dupe_ids[:5]})"
            logging.warning(msg)
            print(f"[~] {msg}")
            warnings += 1

        # Check 8: Gaps in sequence
        if ids:
            sorted_ids = sorted(list(set(ids)))
            # We expect gaps due to 999->1 rollover, but let's check for small gaps
            # Simple check: consecutive
            gaps = []
            for k in range(len(sorted_ids)-1):
                diff = sorted_ids[k+1] - sorted_ids[k]
                if diff > 1 and not (sorted_ids[k] == 999 and sorted_ids[k+1] == 1):
                    # Ignore large jumps if they look like missing data blocks, just flag huge inconsistencies
                    if diff < 50: # Only flag small missing blocks
                        gaps.append((sorted_ids[k], sorted_ids[k+1]))

            if len(gaps) > 0:
                msg = f"Sequence gaps detected (e.g., {gaps[:3]}...)"
                logging.info(msg)
                print(f"[i] {msg}")

        print(f"\nValidation Complete.")
        print(f"Errors: {errors}")
        print(f"Warnings: {warnings}")
        if errors == 0:
            print(">> Data is Clean ✅")
        else:
            print(f">> Check {LOG_FILE} for details ❌")

    except Exception as e:
        print(f"Fatal Validation Error: {e}")

if __name__ == "__main__":
    validate_data()
