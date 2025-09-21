import requests, sys
import pandas as pd
import time
from tqdm import tqdm
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

def build_sequence_csv():
    input_csvs = ['train.csv', 'test.csv']
    script_dir = Path(__file__).resolve().parent
    
    input_ids = []
    for input_csv in input_csvs:
        input_df = pd.read_csv(script_dir / input_csv)
        df = input_df['ensp'].str.split('.', expand=True)
        df.columns = ['ensp_id', 'ensp_ver']
        df.drop_duplicates(inplace=True)
        input_ids.append(df)
    
    unique_pairs = pd.concat(input_ids, ignore_index=True).drop_duplicates()
    sequence_tuples = list()
    for row in tqdm(unique_pairs.itertuples(index=False), total=unique_pairs.shape[0]):
        id = row[0]
        ver = int(row[1])
        time.sleep(0.08)
        protein_info = get_ensembl_id_json(id)
        
        assert(ver == int(protein_info['version']))
        sequence_tuples.append((protein_info['id'], int(protein_info['version']), protein_info['seq']))
                
    df = pd.DataFrame(sequence_tuples, columns=['ensp_id', 'ensp_ver', 'wt_seq'])
    df.to_csv(script_dir / 'Sequences.csv')
    print(f"Saved sequences to {script_dir / 'Sequences.csv'}")

def main():
    build_sequence_csv()

if __name__ == "__main__":
    main()
