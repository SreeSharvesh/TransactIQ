# backend/scripts/build_bias_dataset.py

from datasets import load_dataset
import pandas as pd
import numpy as np
import os

HF_REPO = "sreesharvesh/transactiq-enriched"  # your repo
VAL_ROWS = 4000                               # rows for bias report
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

VAL_PATH = os.path.join(ARTIFACTS_DIR, "val_sample.parquet")

print("Loading HF dataset:", HF_REPO)
ds = load_dataset(HF_REPO)
train = ds["train"]
print("Train size:", len(train))

print("Converting to pandas...")
df = train.to_pandas()

# ---- Normalize column names ----
col_map = {}
if "description" in df.columns:
    col_map["description"] = "transaction_description"
if "merchant_name" in df.columns:
    col_map["merchant_name"] = "merchant"
if "merchant" in df.columns:
    col_map["merchant"] = "merchant"

df = df.rename(columns=col_map)

# Required columns we'll ensure exist:
# transaction_description, category, amount, date, merchant, country

if "transaction_description" not in df.columns:
    raise ValueError("Dataset must have 'description' or 'transaction_description' column.")

if "category" not in df.columns:
    raise ValueError("Dataset must have 'category' column as label.")

if "amount" not in df.columns:
    df["amount"] = np.random.uniform(50, 5000, size=len(df)).round(2)

if "date" not in df.columns:
    df["date"] = pd.date_range("2024-01-01", periods=len(df)).astype(str)

if "merchant" not in df.columns:
    df["merchant"] = df["transaction_description"].str.split().str[0].fillna("UNKNOWN")

if "country" not in df.columns:
    # simple synthetic countries just for bias slicing
    choices = ["AUSTRALIA", "CANADA", "INDIA", "UK", "USA"]
    df["country"] = np.random.choice(choices, size=len(df))

# shuffle once, then take first VAL_ROWS
df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
val_df = df.head(VAL_ROWS).copy()

print("Saving bias/validation sample to:", VAL_PATH)
val_df.to_parquet(VAL_PATH, index=False)
print("Done.")
