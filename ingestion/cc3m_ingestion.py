import pandas as pd
import os
import json

def ingest_cc3m_data(train_tsv: str, val_tsv: str, data_folder: str, output_file: str = "data/processed/cc3m_data.jsonl", sample_size: int = 1000):
    # Load TSVs with correct column order: caption, image_name
    train_df = pd.read_csv(train_tsv, sep="\t", names=["caption", "image_name"], header=None)
    val_df = pd.read_csv(val_tsv, sep="\t", names=["caption", "image_name"], header=None)

    # Remove header row if present
    if train_df.iloc[0]["caption"] == "caption":
        train_df = train_df.iloc[1:]
    if val_df.iloc[0]["caption"] == "caption":
        val_df = val_df.iloc[1:]

    # Combine train and val
    df = pd.concat([train_df, val_df], ignore_index=True)

    # Limit to sample size
    df_sample = df.sample(n=sample_size, random_state=42)

    data_entries = []

    for idx, row in df_sample.iterrows():
        image_name = row["image_name"].strip()
        caption = row["caption"].strip()

        # Determine folder
        image_folder = "training" if os.path.exists(os.path.join(data_folder, "training", image_name)) else "validation"
        image_path = os.path.join(data_folder, image_folder, image_name)

        data_entries.append({
            "id": idx,
            "caption": caption,
            "image_path": image_path
        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data_entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Processed and saved {len(data_entries)} entries to {output_file}")

if __name__ == "__main__":
    ingest_cc3m_data(
        train_tsv="data/cc3m_train.tsv",
        val_tsv="data/cc3m_val.tsv",
        data_folder="data/cc3m"
    )
