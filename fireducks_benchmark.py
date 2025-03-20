import pandas as pd
import modin.pandas as mpd
import ray
import time
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# Initialize Ray
ray.init()

# Generate sample data
num_rows = 10**6
num_cols = 5
data = np.random.randn(num_rows, num_cols)
columns = [f'col_{i}' for i in range(num_cols)]

# Create DataFrames
pandas_df = pd.DataFrame(data, columns=columns)
modin_df = mpd.DataFrame(data, columns=columns)
polars_df = pl.DataFrame(data, schema=columns)

# Performance Comparison
start_time = time.time()
pandas_mean = pandas_df.mean()
pandas_time = time.time() - start_time

start_time = time.time()
modin_mean = modin_df.mean()
modin_time = time.time() - start_time

start_time = time.time()
polars_mean = polars_df.mean()
polars_time = time.time() - start_time

# Display results
print(f"Pandas Time: {pandas_time:.4f}s")
print(f"Modin Time: {modin_time:.4f}s")
print(f"Polars Time: {polars_time:.4f}s")

# Plot comparison
labels = ['Pandas', 'Modin', 'Polars']
times = [pandas_time, modin_time, polars_time]
plt.bar(labels, times, color=['blue', 'green', 'red'])
plt.ylabel("Time (seconds)")
plt.title("Performance Comparison: Pandas vs. Modin vs. Polars")
plt.show()

ray.shutdown()
