import pandas as pd
import csv

# Load your dataset
input_csv = "/home/shash/college/Third_Year/sem6/VR/mini_project2/ourDataset/VQADataset.csv"
output_csv = "VQADatasetPreprocessed.csv"

df = pd.read_csv(input_csv)

# Function to extract first word
def extract_first_word(text):
    if pd.isna(text):
        return ""
    return str(text).strip().split()[0]  # Split and take first word

# Apply transformation
df["answer"] = df["answer"].apply(extract_first_word)

# Save to new CSV with all fields quoted
df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
print(f"✅ Saved processed dataset with single-word answers and quoted fields to: {output_csv}")
