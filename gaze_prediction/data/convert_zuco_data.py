"""Rescale a ZuCo-like word table into the predicted-gaze CSV layout.

Two independent min/max passes:

- nFixations uses its own min/max, then ×100
- FFD, GPT, TRT, GD share *one* min/max across all four, then ×100

That shared range is why predicted FFD and GPT live in the same
numeric ballpark even though raw FFD is ~100 ms and raw GPT is ~200+
ms. Do not treat the output as milliseconds.

Defaults (`training_data/word_averages_v2.csv` →
`sst_et_train_and_vaild_v2.csv`) are relative to this folder and are
not the files checked in next to this script. The idea is what
matters: this is the unit-system break between Track A (z-scores of
ms) and the predicted-gaze files under `gaze_prediction/data/`.
"""

import csv

def scale_value(value, min_val, max_val):
    return ((value - min_val) / (max_val - min_val)) * 100 if max_val > min_val else 0

def convert_csv_format(input_file, output_file):
    # First, determine the min and max values for scaling
    min_nFix, max_nFix = float('inf'), float('-inf')
    min_other, max_other = float('inf'), float('-inf')

    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            nFix = float(row['nFixations'])
            min_nFix, max_nFix = min(min_nFix, nFix), max(max_nFix, nFix)

            for field in ['FFD', 'GPT', 'TRT', 'GD']:
                value = float(row[field])
                min_other, max_other = min(min_other, value), max(max_other, value)

    # Convert and scale the values
    with open(input_file, mode='r', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        # Write header for the new format
        writer.writerow(['sentence_id', 'word_id', 'word', 'nFix', 'FFD', 'GPT', 'TRT', 'GD'])

        for row in reader:
            sentence_id = int(row['Sent_ID'].split('_')[0])
            word_id = int(row['Word_ID'])
            word = row['Word'] if row['Word'] else 'unknown'
            nFix = scale_value(float(row['nFixations']), min_nFix, max_nFix)

            scaled_values = [scale_value(float(row[field]), min_other, max_other) for field in ['FFD', 'GPT', 'TRT', 'GD']]

            # Write transformed and scaled data to the new CSV
            writer.writerow([sentence_id, word_id, word, nFix] + scaled_values)

# Use the function
input_csv = 'training_data/word_averages_v2.csv'
output_csv = 'sst_et_train_and_vaild_v2.csv'
convert_csv_format(input_csv, output_csv)