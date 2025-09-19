import requests

# Example scoreset URN
scoreset_urn = "urn:mavedb:00000069-a-2"

url = f"https://api.mavedb.org/api/v1/score-sets/{scoreset_urn}"

# Send GET request
response = requests.get(url)

# Check if request was successful
if response.ok:
    scoreset_json = response.json()
    print(scoreset_json)
else:
    print(f"Failed to fetch scoreset: {response.status_code}")


# scoreset_json['targetGenes'][0]['targetSequence']['sequenceType']
# scoreset_json['targetGenes'][0]['targetSequence']['sequence']