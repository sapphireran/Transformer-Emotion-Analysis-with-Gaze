"""Build `ZuCo_SST_data/ssts_ZuCo.csv` from per-class .txt folders.

Expects `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt` — those
folders are not in this clone; the output CSV is. Filename stem is
treated as an integer sentence_id so the later join with the gaze
tables can sort 0..399.

Label integers (shared with every other script in this repo):

    NEGATIVE=0  NEUTRAL=1  POSITIVE=2

`ZuCo_SST_data/save_SST_data.py` is the same idea with different
relative paths. Prefer this file's output name (`ssts_ZuCo.csv`).
"""

import os
import pandas as pd

sentiment_mapping = {
    'NEGATIVE': 0,
    'POSITIVE': 2,
    'NEUTRAL': 1
}

data = []

for folder in ['NEGATIVE', 'POSITIVE', 'NEUTRAL']:
    dir_path = os.path.join('ZuCo_SST_data/all', folder)
    for filename in os.listdir(dir_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
            sentence_id = int(os.path.splitext(filename)[0])
            data.append([sentence_id, content, sentiment_mapping[folder]])

df = pd.DataFrame(data, columns=['sentence_id', 'sentence', 'sentiment_label'])

df = df.sort_values(by='sentence_id')

df.to_csv('ZuCo_SST_data/ssts_ZuCo.csv', index=False)