import requests, sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Dict
from io import StringIO


def get_scoreset_json(scoreset_urn) -> Dict:
    # Example scoreset URN
    # scoreset_urn = "urn:mavedb:00000069-a-2"

    url = f"https://api.mavedb.org/api/v1/score-sets/{scoreset_urn}"

    # Send GET request
    response = requests.get(url)

    # Check if request was successful
    if response.ok:
        scoreset_json = response.json()
        return scoreset_json
    else:
        print(f"Failed to fetch scoreset: {response.status_code}")
        return {}


def build_scoreset_csv():
    input_csvs = ['train.csv', 'test.csv']
    script_dir = Path(__file__).resolve().parent

    # collect scoresets across CSVs
    all_ensp = pd.concat(
        [pd.read_csv(script_dir / f)[['scoreset']] for f in input_csvs],
        ignore_index=True
    )
    scoresets = all_ensp['scoreset'].drop_duplicates().tolist()

    # fetch sequences
    sequence_tuples = []
    for scoreset in tqdm(scoresets, desc="Fetching sequences"):
        scoreset_json = get_scoreset_json(scoreset)

        sequence_tuples.append((scoreset, 
                                scoreset_json['targetGenes'][0]['targetSequence']['sequenceType'], 
                                scoreset_json['targetGenes'][0]['targetSequence']['sequence']))

    # save to .csv
    df_out = pd.DataFrame(sequence_tuples, columns=['scoreset', 'target', 'target_seq'])
    output_path = script_dir / 'Info.csv'
    df_out.to_csv(output_path, index=False)
    print(f"Saved protein info to {output_path}")
    

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

def get_TestScore_csv():
    
    input_csv = 'test.csv'
    script_dir = Path(__file__).resolve().parent

    all_ensp = pd.read_csv(script_dir / input_csv)[['scoreset']]

    scoreset_urns = all_ensp['scoreset'].drop_duplicates().tolist()

    all_scores = []

    for urn in scoreset_urns:
        url = f"https://api.mavedb.org/api/v1/score-sets/{urn}/scores"
        response = requests.get(url)
        
        if response.ok:
            data = response.text
            df = pd.read_csv(StringIO(response.text))
            score = df[['accession','score']]
            all_scores.append(score)
        else:
            print(f"Failed to fetch {urn}: {response.status_code}")
    
    if all_scores:
        df_out = pd.concat(all_scores, ignore_index = True)
        output_path = script_dir / 'TestScore.csv'
        df_out.to_csv(output_path, index=False)
        print(f"Saved score info to {output_path}")
    else:
        print('No scores downloaded.')

if __name__ == '__main__':
    get_TestScore_csv()