from pathlib import Path
from data.data_builders import build_scoreset_csv, build_sequence_csv
from scripts.train_esmregressor import train

if __name__ == "__main__":
    # -----------------------------
    # Build CSVs if missing
    # -----------------------------
    if not Path("data/Sequences.csv").exists():
        build_sequence_csv()

    if not Path("data/Info.csv").exists():
        build_scoreset_csv()


    train(
        model_name="facebook/esm2_t6_8M_UR50D",
        batch_size=10,
        epochs=1,
        lr=1e-3,
        weight_decay=1e-2,
        num_workers=4,
        save_name="esmregressor.pt",
    )
