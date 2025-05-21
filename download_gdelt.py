import gdelt
import pandas as pd
from datetime import datetime, timedelta
import os

# Setup
print("[INIT] Initializing GDELT client...")
gd = gdelt.gdelt(version=2)

# Date range
start_date = datetime(2015, 3, 1)
end_date = datetime(2024, 10, 30)
output_file = 'daily_sentiment.csv'

# Load checkpoint if exists
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    processed_days = set(existing_df['Date'])
    print(f"[RESUME] Found existing file with {len(processed_days)} days processed.")
else:
    existing_df = pd.DataFrame()
    processed_days = set()
    print("[START] No previous results found, starting fresh.")

# Processing loop
current = start_date
while current <= end_date:
    day_str = current.strftime('%Y-%m-%d')

    # Skip weekends
    if current.weekday() >= 5:
        current += timedelta(days=1)
        continue

    if day_str in processed_days:
        print(f"[SKIP] Day already processed: {day_str}")
        current += timedelta(days=1)
        continue

    try:
        print(f"\n[INFO] Fetching data for {day_str}...")
        next_day_str = (current + timedelta(days=1)).strftime('%Y-%m-%d')
        df = gd.Search([day_str, next_day_str], table='gkg')

        if df.empty:
            print("[WARN] No data returned.")
            result = {
                'Date': day_str,
                'Tone': None,
                'Positive_Score': None,
                'Negative_Score': None,
                'Polarity': None,
                'Activity_Ref_Density': None,
                'Self_Group_Density': None
            }
        else:
            print(f"[INFO] Downloaded {len(df)} rows.")
            print("[INFO] Splitting and parsing V2Tone...")

            tone_cols = df['V2Tone'].str.split(',', expand=True)
            tone_cols = tone_cols.iloc[:, :6]
            tone_cols.columns = [
                'Tone', 'Positive_Score', 'Negative_Score',
                'Polarity', 'Activity_Ref_Density', 'Self_Group_Density'
            ]
            tone_cols = tone_cols.apply(pd.to_numeric, errors='coerce')
            tone_means = tone_cols.mean()

            print(f"[INFO] Averages: {tone_means.to_dict()}")

            result = {'Date': day_str, **tone_means.to_dict()}

        # Append result to CSV
        pd.DataFrame([result]).to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
        print(f"[CHECKPOINT] Saved results for {day_str}.")

    except Exception as e:
        print(f"[ERROR] Failed for {day_str}: {e}")

    current += timedelta(days=1)

print("\n[DONE] All weekdays processed and saved.")
