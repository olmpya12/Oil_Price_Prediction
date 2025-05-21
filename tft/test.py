import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("/home/ilayda/Workspace/oil_price/data/processed/filtered_merged_dataset_trimmed.csv", parse_dates=["Date"])
df = df.sort_values("Date")

# Display basic structure
print("✅ Dataset loaded successfully")
print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"🧾 Number of rows: {len(df)}")
print(f"📊 Columns:\n{df.columns.tolist()}")

# Check missing values
missing = df.isna().sum()
print("\n🔍 Missing values per column:")
print(missing[missing > 0])

# Plot crude oil price
plt.figure(figsize=(12, 4))
plt.plot(df["Date"], df["Crude Oil ($/barrel)"], label="Crude Oil ($/barrel)")
plt.title("Crude Oil Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Check daily frequency gaps
df["delta"] = df["Date"].diff().dt.days
gap_counts = df["delta"].value_counts().sort_index()
print("\n⏱ Frequency of time gaps (in days):")
print(gap_counts)

# Drop helper column
df.drop(columns="delta", inplace=True)

# Show first few rows
print("\n🔎 First few rows:")
print(df.head())
