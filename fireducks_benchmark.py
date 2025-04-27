# Set Modin engine before importing modin
import os
os.environ["MODIN_ENGINE"] = "ray"

# Install dependencies
!pip install modin[ray] ray polars fireducks matplotlib

# Imports
import pandas as pd
import modin.pandas as mpd
import polars as pl
import numpy as np
import ray
import time
import matplotlib.pyplot as plt

import fireducks
from fireducks.core import get_fireducks_options

# ── FireDucks “Benchmark Mode” Setup ────────────────────────────────────────────
# 1) Grab the options object
fireducks_options = get_fireducks_options()
# 2) Enable benchmark mode
fireducks_options.set_benchmark_mode(True)
# 3) Helper to force any lazy computation
def evaluate(df):
    try:
        df._evaluate()
    except AttributeError:
        pass
# ────────────────────────────────────────────────────────────────────────────────

# Sample data creation
num_rows = 10**6
num_cols = 5
data = np.random.randn(num_rows, num_cols)
columns = [f'col_{i}' for i in range(num_cols)]

# Create DataFrames
pandas_df = pd.DataFrame(data, columns=columns)
modin_df = mpd.DataFrame(data, columns=columns)
polars_df = pl.DataFrame(data, schema=columns)

# Benchmark helper
def benchmark_mean(df):
    start = time.time()
    result = df.mean()
    evaluate(df)
    return time.time() - start

# Run benchmarks
pandas_time     = benchmark_mean(pandas_df)
modin_time      = benchmark_mean(modin_df)
polars_time     = benchmark_mean(polars_df)
fireducks_time  = benchmark_mean(pandas_df)  # FireDucks instruments pandas_df

# Print results
print(f"Pandas Time:            {pandas_time:.4f} seconds")
print(f"Modin Time:             {modin_time:.4f} seconds")
print(f"Polars Time:            {polars_time:.4f} seconds")
print(f"FireDucks (Pandas) Time:{fireducks_time:.4f} seconds")

# Plot results
labels = ['Pandas', 'Modin', 'Polars', 'FireDucks (Pandas)']
times  = [pandas_time, modin_time, polars_time, fireducks_time]

plt.figure(figsize=(8, 6))
plt.bar(labels, times, color=['blue', 'green', 'red', 'orange'])
plt.ylabel("Time (seconds)")
plt.title("Performance Comparison: Pandas vs. Modin vs. Polars vs. FireDucks")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
