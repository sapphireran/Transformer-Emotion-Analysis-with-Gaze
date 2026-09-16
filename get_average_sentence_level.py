"""Average 12 per-subject sentence CSVs, then min-max and z-score.

Reads `et_csv_data/{1-12}_SR.csv` (note: git has `ZuCo_et_csv_data/`).
Zeros on non-id columns are treated as missing before the mean, so a
subject who skipped a region does not drag that region's average to 0.

Averaging is `groupby(level=0).mean()` — i.e. by *row position*, not
by sentence id. That is only safe if every subject file has the same
400 rows in the same order. Task-1 subject 2 does not (see
`DataTransformer` skips). The checked-in `average_data.csv` has 400
rows; do not blindly re-run this against the checked-in `*_SR.csv`
files and assume you will reproduce it.

Outputs (written next to the inputs):

    min_max_scaled_average_data.csv
    standard_scaled_average_data.csv

`standard` is what `model_ZuCo_SST.py` trains on after the labels are
joined. This script does not write `average_data.csv` itself — that
raw mean is an intermediate you keep if you add a `to_csv` for
`average_df`.
"""

import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 文件夹路径
# Trap: checked-in folder is ZuCo_et_csv_data/.
folder_path = 'et_csv_data'

# 读取所有 CSV 文件
dataframes = []
for i in range(0, 12):
    file_path = os.path.join(folder_path, f"{i + 1}_SR.csv")
    df = pd.read_csv(file_path)
    # 选择除第 0 列以外的所有列
    columns_except_first = df.columns[1:]
    # 只将这些列中的 0 值替换为 NaN
    df[columns_except_first] = df[columns_except_first].replace(0, np.nan)
    dataframes.append(df)

# 计算平均值
average_df = pd.concat(dataframes).groupby(level=0).mean()

# 将索引转换为 'id' 列，并转换为整数
average_df.reset_index(inplace=True)
average_df['id'] = average_df['id'].astype(int)

# 分离 'id' 列
id_column = average_df['id']
average_df_no_id = average_df.drop(columns=['id'])

# 确保 id_column 是一维的
id_column = id_column.values

# Min-Max Scaling
min_max_scaler = MinMaxScaler()
scaled_data_min_max = min_max_scaler.fit_transform(average_df_no_id)
scaled_average_df_min_max = pd.DataFrame(scaled_data_min_max, columns=average_df_no_id.columns)

# 将 'id' 列设置为索引
scaled_average_df_min_max['id'] = id_column
scaled_average_df_min_max.set_index('id', inplace=True)

# 保存 Min-Max Scaling 结果
scaled_average_df_min_max.to_csv(os.path.join(folder_path, 'min_max_scaled_average_data.csv'))

# Standard Scaling
standard_scaler = StandardScaler()
scaled_data_standard = standard_scaler.fit_transform(average_df_no_id)
scaled_average_df_standard = pd.DataFrame(scaled_data_standard, columns=average_df_no_id.columns)

# 将 'id' 列设置为索引
scaled_average_df_standard['id'] = id_column
scaled_average_df_standard.set_index('id', inplace=True)

# 保存 Standard Scaling 结果
scaled_average_df_standard.to_csv(os.path.join(folder_path, 'standard_scaled_average_data.csv'))