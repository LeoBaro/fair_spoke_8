from pathlib import Path
from fastparquet import ParquetFile
import matplotlib.pyplot as plt
import pandas as pd

parquet_file = ParquetFile("/home/leobaro/Downloads/datasets/web/datacomp/shards/00000000.parquet")
print(parquet_file.columns)


path = Path("/home/leobaro/Downloads/datasets/web/datacomp/metadata")
# file = "00a735dd2dd539edc0f4368bd13a0c37.parquet"

parquet_files = list(path.glob("*.parquet"))

columns_to_load = ['clip_b32_similarity_score', 'clip_l14_similarity_score']

data_frames = []

for file in parquet_files:
    parquet_file = ParquetFile(file)
    df = parquet_file.to_pandas(columns=columns_to_load)
    data_frames.append(df)  #

df = pd.concat(data_frames, ignore_index=True)

print(df.info())
# # Open Parquet file and inspect metadata
# parquet_file = ParquetFile(path / file)
# print(parquet_file.info)            # Schema and row groups
# print(parquet_file.columns)         # Column names

columns_to_load = ['clip_b32_similarity_score', 'clip_l14_similarity_score']

plt.figure(figsize=(12, 6))

# Histogram for clip_b32_similarity_score
plt.subplot(1, 2, 1)
df['clip_b32_similarity_score'].hist(
    bins=30, color='skyblue', 
    edgecolor='black', alpha=0.5, label="clip_b32_similarity_score"
)
plt.title('clip_b32_similarity_score Histogram')
plt.xlabel('Score')
plt.ylabel('Frequency')

# Histogram for clip_l14_similarity_score
df['clip_l14_similarity_score'].hist(
    bins=30, color='salmon', edgecolor='black', alpha=0.5, label="clip_l14_similarity_score"
)
plt.title('clip_l14_similarity_score Histogram')
plt.xlabel('Score')
plt.ylabel('Frequency')

plt.grid(alpha=0.1)
plt.legend()
# Show the plots
plt.tight_layout()
plt.show()