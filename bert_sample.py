import json
import requests

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/msmarco-distilbert-base-tas-b"
headers = {"Authorization": f"Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


data = query(
    {
        "inputs": {
            "source_sentence": "That is a happy person",
            "sentences": [
                "That is a happy dog",
                "That is a very happy person",
                "Today is a sunny day",
            ],
        }
    }
)
## [0.853, 0.981, 0.655]
print(data)

import pandas as pd
import math

synonyms = pd.read_csv("synonyms.csv")

groups = {}
import tqdm

for i, row in tqdm.tqdm(synonyms.iterrows()):
    if not isinstance(row["synonyms"], float):
        groups[row["lemma"]] = row["synonyms"].split(";")

for i, row in synonyms.iterrows():
    print(row)
    break
