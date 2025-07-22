import pandas as pd

df = pd.read_csv("/Users/shivampatel/Research/Chest_X_Ray/chest_xray/chest_xray_predictions.csv")

print(df.columns.tolist())

print(df['true_label'].value_counts())