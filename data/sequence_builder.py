import requests, sys
import pandas as pd
import time
from tqdm import tqdm
import argparse
from pathlib import Path

def get_ensembl_id_json(ensembl_id: str):
    server = "https://rest.ensembl.org"
    ext = f"/sequence/id/{ensembl_id}"
 
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    
    decoded = r.json()
    return decoded

def main(split: str):
    # Get the directory of the current script
    script_dir = Path(__file__).resolve().parent

    # Construct path to the CSV
    csv_path = script_dir / f'{split}.csv'
    
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        sys.exit(1)

    train_set = pd.read_csv(csv_path)

    id_ver_pairs = set()
    sequence_tuples = list()

    for id, version in tqdm(train_set['ensp'].str.split('.')):
        if (id, int(version)) not in id_ver_pairs:
            time.sleep(0.08)
            protein_info = get_ensembl_id_json(id)
            sequence_tuples.append((protein_info['id'], int(protein_info['version']), protein_info['seq']))
            
            id_ver_pairs.add((id, int(version)))
            
    df = pd.DataFrame(sequence_tuples, columns=['ensp_id', 'ensp_ver', 'wt_seq'])
    df.set_index(['ensp_id', 'ensp_ver'], inplace=True)

    output_path = script_dir / f'{split}_seq.csv'
    df.to_csv(output_path)
    print(f"Saved sequences to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Ensembl sequences for train or test split.")
    parser.add_argument("split", choices=["train", "test"], help="Which dataset split to process")
    args = parser.parse_args()
    
    main(args.split)
