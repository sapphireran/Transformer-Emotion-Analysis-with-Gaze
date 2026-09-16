"""12-reader mean of the word-level `*_SR.csv` files in this folder.

Averages the seven numeric gaze columns by row index, then pastes
`id, Sent_ID, Word_ID, Word, WordLen` from `1_SR.csv`. Empty `Word`
cells become `unknown`; remaining numeric NaNs become 0.

The 0→NaN replace is commented out on purpose in the original (see
the line below). That is a *different* missing-data policy than
`get_average_sentence_level.py`, which does treat 0 as missing.
`word_averages_v2.csv` is the output this clone treats as canonical.
"""

import pandas as pd
import numpy as np
import os

# 要计算平均值的列
columns_to_average = ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT']

# 读取每个CSV文件并计算平均值
dataframes = []
for i in range(12):
    file_path = os.path.join(f"{i + 1}_SR.csv")
    df = pd.read_csv(file_path)

    # 只将这些列中的 0 值替换为 NaN
    #df[columns_to_average] = df[columns_to_average].replace(0, np.nan)

    dataframes.append(df)

# 计算平均值
average_df = pd.concat(dataframes).groupby(level=0).mean()[columns_to_average]

# 从第一个文件中提取固定列
first_file_path = os.path.join("1_SR.csv")
first_file_df = pd.read_csv(first_file_path)
fixed_columns = first_file_df[['id', 'Sent_ID', 'Word_ID', 'Word', 'WordLen']]

# 合并固定列和平均值列
combined_df = pd.concat([fixed_columns, average_df], axis=1)

# 替换空值：'word'列的null变成'unknown'，其他列的null变成0
combined_df['Word'] = combined_df['Word'].fillna('unknown')
combined_df[columns_to_average] = combined_df[columns_to_average].fillna(0)

# 重新安排列的顺序
final_columns_order = ['id', 'Sent_ID', 'Word_ID', 'Word'] + columns_to_average + ['WordLen']
combined_df = combined_df[final_columns_order]

# 保存到新的CSV文件
combined_df.to_csv(os.path.join("word_averages_v2.csv"), index=False)