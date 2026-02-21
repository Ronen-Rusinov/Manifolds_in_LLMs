"""
"Why aren't there direct methods to load only the activations????"
Cause you're going to lose the correct links between the prompts and
the activations data. Any shuffling will be on the entire pandas dataframe, this is a failsafe
which should not be circumvented.
Also, this script should be loaded in its entirety, not just 'from . import DATAPATH". Weird behaviour might happen.
"""

import os
from pathlib import Path
import pandas as pd
import time

#PLEASE NOTE THOSE TWO ARE NOT STRINGS. If you want to pass them as arguments, stringify them first, will save you a headache.

LLM_PROJ_PATH = os.environ.get("LLM_PROJ_PATH")
if LLM_PROJ_PATH:
    DATAPATH = Path(LLM_PROJ_PATH) / "data" / "activations_data"
else:
    DATAPATH = Path(__file__).parent.parent.parent / "data" / "activations_data"

FIRST_PARQUET_PATH = DATAPATH / "activations_part_01_of_10.parquet"
def load_first_parquet(timing=False):
    if not FIRST_PARQUET_PATH.exists():
        raise FileNotFoundError(f"First parquet file not found at {FIRST_PARQUET_PATH}. Please run pull_data.py to download the data.")
    start_time = time.time()
    df = pd.read_parquet(FIRST_PARQUET_PATH)
    if timing:
        print(f"Time to load first parquet: {time.time() - start_time}")
    return df

def load_all_parquets(timing=False):
    parquet_files = sorted(DATAPATH.glob("activations_part_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {DATAPATH}. Please run pull_data.py to download the data.")
    dataframes = []
    for parquet_file in parquet_files:
        print(f"Loading {parquet_file}...")
        start_time = time.time()
        df = pd.read_parquet(parquet_file)
        if timing:
            print(f"Time to load {parquet_file}: {time.time() - start_time}")
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def load_train_test_first_parquet(train_size= 0.8, timing=False):
    if not FIRST_PARQUET_PATH.exists():
        raise FileNotFoundError(f"First parquet file not found at {FIRST_PARQUET_PATH}. Please run pull_data.py to download the data.")
    start_time = time.time()
    df = pd.read_parquet(FIRST_PARQUET_PATH)
    if timing:
        print(f"Time to load first parquet: {time.time() - start_time}")
    shuffled_df = df.sample(frac=1, random_state=42)
    train_df = shuffled_df.iloc[:int(train_size * len(shuffled_df))]
    test_df = shuffled_df.iloc[int(train_size * len(shuffled_df)):]
    if timing:
        print(f"Time to split train/test: {time.time() - start_time}")
    return train_df, test_df

def load_train_test_all_parquets(train_size= 0.8, timing=False):
    parquet_files = sorted(DATAPATH.glob("activations_part_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {DATAPATH}. Please run pull_data.py to download the data.")
    dataframes = []
    for parquet_file in parquet_files:
        print(f"Loading {parquet_file}...")
        start_time = time.time()
        df = pd.read_parquet(parquet_file)
        if timing:
            print(f"Time to load {parquet_file}: {time.time() - start_time}")
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    shuffled_df = combined_df.sample(frac=1, random_state=42)
    train_df = shuffled_df.iloc[:int(train_size * len(shuffled_df))]
    test_df = shuffled_df.iloc[int(train_size * len(shuffled_df)):]
    if timing:
        print(f"Time to split train/test: {time.time() - start_time}")
    return train_df, test_df

def load_train_test_val_first_parquet(train_size= 0.7, val_size= 0.2, timing=False):
    if not FIRST_PARQUET_PATH.exists():
        raise FileNotFoundError(f"First parquet file not found at {FIRST_PARQUET_PATH}. Please run pull_data.py to download the data.")
    start_time = time.time()
    df = pd.read_parquet(FIRST_PARQUET_PATH)
    if timing:
        print(f"Time to load first parquet: {time.time() - start_time}")
    shuffled_df = df.sample(frac=1, random_state=42)
    train_end = int(train_size * len(shuffled_df))
    val_end = train_end + int(val_size * len(shuffled_df))
    train_df = shuffled_df.iloc[:train_end]
    val_df = shuffled_df.iloc[train_end:val_end]
    test_df = shuffled_df.iloc[val_end:]
    if timing:
        print(f"Time to split train/val/test: {time.time() - start_time}")
    return train_df, val_df, test_df

def load_train_test_val_all_parquets(train_size= 0.7, val_size= 0.2, timing=False):
    parquet_files = sorted(DATAPATH.glob("activations_part_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {DATAPATH}. Please run pull_data.py to download the data.")
    dataframes = []
    for parquet_file in parquet_files:
        print(f"Loading {parquet_file}...")
        start_time = time.time()
        df = pd.read_parquet(parquet_file)
        if timing:
            print(f"Time to load {parquet_file}: {time.time() - start_time}")
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    shuffled_df = combined_df.sample(frac=1, random_state=42)
    train_end = int(train_size * len(shuffled_df))
    val_end = train_end + int(val_size * len(shuffled_df))
    train_df = shuffled_df.iloc[:train_end]
    val_df = shuffled_df.iloc[train_end:val_end]
    test_df = shuffled_df.iloc[val_end:]
    if timing:
        print(f"Time to split train/val/test: {time.time() - start_time}")
    return train_df, val_df, test_df

if __name__ == "__main__":
    #speed test
    import time
    df = load_first_parquet(timing=True)
    #print fields and dtypes
    print(df.dtypes)

    