from pathlib import Path
from data.data_builders import build_scoreset_csv, build_sequence_csv
from utils.train_eval import train_model

if __name__ == "__main__":
    # -----------------------------
    # Build CSVs if missing
    # -----------------------------
    if not Path("data/Sequences.csv").exists():
        build_sequence_csv()

    if not Path("data/Info.csv").exists():
        build_scoreset_csv()


    train_model(
        model_name="OneHotRegressor",
        batch_size=1,
        epochs=1,
        lr=1e-3,
        weight_decay=1e-2,
        num_workers=4,
        data_dir="data",
        save_name="esmregressor.pt",
        dataset_encoding="one-hot-segment"
    )
