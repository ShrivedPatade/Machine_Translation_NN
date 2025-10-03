# File: preprocess.py (Final Corrected Version)

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- FINAL CORRECT IMPORT, BASED ON YOUR SYSTEM'S OUTPUT ---
from sanskrit_parser import Parser

# --- 1. Sandhi Splitting Function (Final Version) ---
def apply_sandhi_splitting(df, column_name='Sanskrit'):
    """Applies Sandhi splitting using the sanskrit-parser library."""
    print("Initializing Sanskrit parser...")
    # --- FINAL CORRECT INSTANTIATION ---
    parser = Parser()
    
    print("Applying Sandhi splitting... This can take several minutes.")
    
    def split_sentence(sentence):
        """Wrapper function for a single sentence."""
        if not isinstance(sentence, str) or not sentence:
            return sentence
        try:
            split_string = parser.split_sandhi(sentence)
            return str(split_string) # Ensure output is a string
        except Exception as e:
            return sentence

    df[column_name] = df[column_name].apply(split_sentence)
    
    print("Sandhi splitting complete.")
    return df

# --- 2. Data Splitting Function (Unchanged) ---
def split_data(df):
    """Splits the DataFrame into train, dev, and test sets."""
    train, temp = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)
    dev, test = train_test_split(temp, test_size=0.5, random_state=42, shuffle=True)
    return train, dev, test

# --- 3. File Writing Function (Unchanged) ---
def write_to_files(df, split_name, path):
    """Writes the English and Sanskrit columns of a DataFrame to separate text files."""
    en_path = os.path.join(path, f"{split_name}.en")
    sa_path = os.path.join(path, f"{split_name}.sa")
    
    df['English'].to_csv(en_path, index=False, header=False, encoding='utf-8')
    df['Sanskrit'].to_csv(sa_path, index=False, header=False, encoding='utf-8')
    print(f"Wrote {split_name}.en and {split_name}.sa")


# ===================================================================
# Main execution block (Unchanged)
# ===================================================================
if __name__ == "__main__":
    
    base_path = '../parallel-corpus'
    input_path = os.path.join(base_path, 'data', 'raw_corpus.csv')
    output_path = os.path.join(base_path, 'data', 'final_data')

    os.makedirs(output_path, exist_ok=True)
    
    print(f"Reading consolidated data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} raw pairs.")

    df['English'] = df['English'].str.lower().str.strip()
    df['Sanskrit'] = df['Sanskrit'].str.strip()
    df.dropna(subset=['English', 'Sanskrit'], inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"Pairs after cleaning and dropping duplicates: {len(df)}")
    
    df = apply_sandhi_splitting(df)

    train_df, dev_df, test_df = split_data(df)
    print(f"Data split: {len(train_df)} train, {len(dev_df)} dev, {len(test_df)} test.")

    write_to_files(train_df, split_name='train', path=output_path)
    write_to_files(dev_df, split_name='dev', path=output_path)
    write_to_files(test_df, split_name='test', path=output_path)
    
    print("\n----------------------------------------------------")
    print("Preprocessing complete!")
    print(f"Final files are ready in: {output_path}")
    print("----------------------------------------------------")