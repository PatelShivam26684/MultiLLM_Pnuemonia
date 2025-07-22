import pandas as pd

df = pd.read_csv("/Users/shivampatel/Research/Chest_X_Ray/chest_xray/chest_xray_predictions.csv")

# Clean all response columns: convert to string and strip whitespace
response_cols = [
    'gpt_4o_response', 'gpt_41_response',
    'gemini_response', 'claude_response', 'grok_response'
]

for col in response_cols:
    df[col] = df[col].astype(str).str.strip()

# Verify value counts for each column
for col in response_cols:
    print(f"\n--- Value counts for {col} ---")
    print(df[col].value_counts(dropna=False))

# Create filter condition
condition = True
for col in response_cols:
    condition = condition & df[col].isin(['0', '1'])

df_clean = df[condition]

print(f"\nFound {len(df_clean)} valid rows")

# Convert to integers if we have valid rows
if not df_clean.empty:
    df_clean[response_cols] = df_clean[response_cols].astype(int)
    print(df_clean.head())
else:
    print("No rows satisfy the condition '0' or '1' in all response columns")


# Save to a new CSV
#df_clean.to_csv("cleaned_file.csv", index=False)

df_clean.to_csv("/Users/shivampatel/Research/Chest_X_Ray/chest_xray/chest_xray_predictions.csv", index=False)