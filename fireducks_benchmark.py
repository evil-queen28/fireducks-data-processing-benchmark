import os
os.environ["MODIN_ENGINE"] = "ray"  # Set before importing modin

# Install dependencies
!pip install modin[ray] ray polars fireducks

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

# Init FireDucks Benchmark Mode
get_fireducks_options().set_benchmark_mode(True)


# Sample data
num_rows = 10**6
num_cols = 5
data = np.random.randn(num_rows, num_cols)
columns = [f'col_{i}' for i in range(num_cols)]

# Create DataFrames
pandas_df = pd.DataFrame(data, columns=columns)
modin_df = mpd.DataFrame(data, columns=columns)
polars_df = pl.DataFrame(data, schema=columns)

# Evaluate function
def evaluate(df):
    try:
        df._evaluate()
    except AttributeError:
        pass

# Benchmarking
start_time = time.time()
pandas_mean = pandas_df.mean()
pandas_time = time.time() - start_time

start_time = time.time()
modin_mean = modin_df.mean()
modin_time = time.time() - start_time

start_time = time.time()
polars_mean = polars_df.mean()
polars_time = time.time() - start_time

# FireDucks works by automatically instrumenting Pandas/Modin
start_time = time.time()
fireducks_mean = pandas_df.mean()  # This is tracked since benchmark mode is ON
evaluate(pandas_df)
fireducks_time = time.time() - start_time

# Results
print(f"Pandas Time: {pandas_time:.4f}s")
print(f"Modin Time: {modin_time:.4f}s")
print(f"Polars Time: {polars_time:.4f}s")
print(f"FireDucks (Pandas) Time: {fireducks_time:.4f}s")

# Plot
labels = ['Pandas', 'Modin', 'Polars', 'FireDucks (Pandas)']
times = [pandas_time, modin_time, polars_time, fireducks_time]
plt.bar(labels, times, color=['blue', 'green', 'red', 'orange'])
plt.ylabel("Time (seconds)")
plt.title("Performance Comparison: Pandas vs. Modin vs. Polars vs. FireDucks")
plt.show
