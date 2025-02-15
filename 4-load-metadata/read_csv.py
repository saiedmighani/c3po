import pandas as pd
import os

def split_csv_into_batches(input_csv, output_dir, batch_size=1000):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(input_csv, chunksize=batch_size)
    df = (chunk.dropna(subset=["description", "name"]) for chunk in df)
    df = (chunk[(chunk["description"].str.strip() != "") & (chunk["name"].str.strip() != "")] for chunk in df)

    # Write each batch to a separate CSV file
    for i, chunk in enumerate(df):
        batch_file = os.path.join(output_dir, f"batch_{i+1}.csv")
        chunk.to_csv(batch_file, index=False)
        print(f"Saved: {batch_file}")

# Example usage
input_csv = "./cleaned_dataset_no_duplicates.csv"  # Replace with your actual file
output_dir = "./batch_files/"
split_csv_into_batches(input_csv, output_dir)
