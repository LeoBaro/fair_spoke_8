# How to download datacomp shards

1. Backup the data of previous shard x:
   1. shards -> _shards_x 
   2. metadata -> _metadata_x
2. Create new empty folders:
   1. mkdir shards
   2. mkdir metadata
3. Copy the corresponding metadata file inside the metadata folder:
   1. cp all_metadata_backup/005e1941475026c5167278c9b564d204.parquet metadata
4. Start the download bash script