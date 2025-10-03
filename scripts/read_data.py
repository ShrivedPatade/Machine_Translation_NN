# File: prepare_corpus.py
# Purpose: Reads data from parallel text files and consolidates it into a single raw CSV file.

import pandas as pd
import os

# ===================================================================
# A single, reusable function to read parallel text files
# ===================================================================
def read_parallel_files(en_path, sa_path):
    """
    Reads parallel English and Sanskrit files line by line and returns a DataFrame.
    This handles both .sa and .sn extensions for Sanskrit.
    """
    eng_lines, sa_lines = [], []
    try:
        with open(en_path, 'r', encoding='utf-8') as f:
            eng_lines = [line.strip() for line in f.readlines()]
        
        with open(sa_path, 'r', encoding='utf-8') as f:
            sa_lines = [line.strip() for line in f.readlines()]

        if len(eng_lines) != len(sa_lines):
            print(f"Warning: Mismatch in line numbers for {en_path} ({len(eng_lines)}) and {sa_path} ({len(sa_lines)}). Truncating to shorter length.")
            min_len = min(len(eng_lines), len(sa_lines))
            eng_lines = eng_lines[:min_len]
            sa_lines = sa_lines[:min_len]

        return pd.DataFrame({'English': eng_lines, 'Sanskrit': sa_lines})

    except FileNotFoundError as e:
        print(f"Warning: Could not find files for a pair. Error: {e}. Skipping.")
        return pd.DataFrame() # Return an empty DataFrame if files are not found

# ===================================================================
# Main execution block
# ===================================================================

if __name__ == "__main__":
    print("Starting data consolidation from parallel text files...")
    
    # Define the base path to your corpus
    base_path = '../parallel-corpus/data/' # Adjust if your script is in a different location

    # List of all data sources based on your screenshot
    # Each item is a tuple: (folder_name, english_file, sanskrit_file)
    data_sources = [
        ('bible', 'bible.en', 'bible.sa'),
        ('gitasopanam', 'gitasopanam.en', 'gitasopanam.sa'),
        ('itihasa/data', 'train.en', 'train.sn'), # Note the .sn extension
        ('itihasa/data', 'test.en', 'test.sn'),   # Including test set from itihasa
        ('itihasa/data', 'dev.en', 'dev.sn'),     # Including dev set from itihasa
        ('mkb', 'mkb.en', 'mkb.sa'),
        ('nios', 'nios.en', 'nios.sa'),
        ('spoken-tutorials', 'spoken.en', 'spoken.sa')
    ]

    # A list to hold all the dataframes we load
    all_dataframes = []

    for folder, en_file, sa_file in data_sources:
        print(f"Reading from {folder}...")
        en_full_path = os.path.join(base_path, folder, en_file)
        sa_full_path = os.path.join(base_path, folder, sa_file)
        
        df = read_parallel_files(en_full_path, sa_full_path)
        if not df.empty:
            all_dataframes.append(df)
            print(f" -> Loaded {len(df)} pairs.")

    # Concatenate all loaded dataframes into one
    full_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Define the output path for the raw, consolidated data
    output_path = os.path.join(base_path, 'raw_corpus.csv')
    
    # Save the dataframe to a CSV file
    full_df.to_csv(output_path, index=False)
    
    print("\n----------------------------------------------------")
    print("Consolidation complete.")
    print(f"Total sentence pairs collected: {len(full_df)}")
    print(f"Raw consolidated data saved to: {output_path}")
    print("----------------------------------------------------")