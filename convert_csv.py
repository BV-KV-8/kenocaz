import csv
import json
import pandas as pd

def convert_old_to_new(input_file='games_old.csv', output_file='games.csv'):
    """
    Convert old CSV format (draw_id, timestamp, [numbers])
    to new format (draw_id, timestamp, n1, n2, ..., n20)
    """
    print(f"Converting {input_file} to new format...")

    old_data = []

    # Read old format
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                try:
                    draw_id = row[0]
                    timestamp = row[1]
                    numbers = json.loads(row[2])

                    if len(numbers) == 20:
                        old_data.append({
                            'draw_id': draw_id,
                            'timestamp': timestamp,
                            'numbers': sorted(numbers)
                        })
                except:
                    continue

    print(f"Loaded {len(old_data)} valid draws")

    # Write new format
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['draw_id', 'timestamp'] + [f'n{i+1}' for i in range(20)]
        writer.writerow(header)

        # Data rows
        for draw in old_data:
            row = [draw['draw_id'], draw['timestamp']] + draw['numbers']
            writer.writerow(row)

    print(f"✓ Saved {len(old_data)} draws to {output_file}")

    # Verify
    df = pd.read_csv(output_file)
    print(f"\nVerification:")
    print(f"  Columns: {len(df.columns)} (should be 22)")
    print(f"  Rows: {len(df)}")
    print(f"\nSample (first row):")
    print(df.head(1))

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'games.csv'
        convert_old_to_new(input_file, output_file)
    else:
        print("Usage: python convert_csv.py <old_file> [new_file]")
        print("\nConverting default: games_old.csv -> games.csv")

        import os
        if os.path.exists('games_old.csv'):
            convert_old_to_new()
        elif os.path.exists('games.csv'):
            # Check if current games.csv is old format
            with open('games.csv', 'r') as f:
                first_line = f.readline()
                if '"[' in first_line:  # Old format detected
                    print("Old format detected in games.csv")
                    print("Backing up to games_backup.csv...")
                    os.rename('games.csv', 'games_backup.csv')
                    convert_old_to_new('games_backup.csv', 'games.csv')
                else:
                    print("games.csv already in new format!")
        else:
            print("No input file found!")
