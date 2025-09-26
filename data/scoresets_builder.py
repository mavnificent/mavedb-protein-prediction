import requests
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Dict

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


    # scoreset_json['targetGenes'][0]['targetSequence']['sequenceType']
    # scoreset_json['targetGenes'][0]['targetSequence']['sequence']


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

def main():
    build_scoreset_csv()

if __name__ == "__main__":
    main()
