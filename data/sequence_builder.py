import requests, sys
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path

def get_ensembl_id_json(ensembl_id: str):
    server = "https://rest.ensembl.org"
    ext = f"/sequence/id/{ensembl_id}"
 
    r = requests.get(server+ext, headers={"Content-Type": "application/json"})
    
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    
    decoded = r.json()
    return decoded

def build_sequence_csv():
    input_csvs = ['train.csv', 'test.csv']
    script_dir = Path(__file__).resolve().parent

    # collect unique ENSP IDs across CSVs
    all_ensp = pd.concat(
        [pd.read_csv(script_dir / f)[['ensp']] for f in input_csvs],
        ignore_index=True
    )
    unique_ids = all_ensp['ensp'].drop_duplicates().tolist()

    # fetch sequences
    sequence_tuples = []
    for ensp in tqdm(unique_ids, desc="Fetching sequences"):
        if '.' not in ensp:
            print(f"Skipping unexpected format: {ensp}")
            continue

        ensp_id, ver = ensp.split('.')
        protein_info = get_ensembl_id_json(ensp_id)

        assert int(ver) == int(protein_info['version']), f"Version mismatch for {ensp}"

        sequence_tuples.append((ensp, protein_info['seq']))

    # save to .csv
    df_out = pd.DataFrame(sequence_tuples, columns=['ensp', 'wt_seq'])
    output_path = script_dir / 'Sequences.csv'
    df_out.to_csv(output_path, index=False)
    print(f"Saved sequences to {output_path}")

def main():
    build_sequence_csv()

if __name__ == "__main__":
    main()
