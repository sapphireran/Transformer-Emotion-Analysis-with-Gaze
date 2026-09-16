"""80/10/10 split of the 400 ZuCo rows. Not used by model_ZuCo_SST.py.

The training script runs StratifiedKFold on the combined table instead.
These files (`train.csv` 320, `valid.csv` 40, `test.csv` 40) are a
convenience split (`random_state=42`, not stratified). The 40-row
valid slice is *not* balanced (7/14/19 in this clone) — do not quote
it as the official test.

Same typo as `SST_data/spilt.py`.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
file_path = 'combined_sst_et_standard.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 拆分数据集为训练集和（验证集+测试集），比例为 80%:20%
train_df, valid_test_df = train_test_split(df, test_size=0.2, random_state=42)

# 将（验证集+测试集）拆分为验证集和测试集，各占 50%
valid_df, test_df = train_test_split(valid_test_df, test_size=0.5, random_state=42)

# 保存为新的 CSV 文件
train_df.to_csv('train.csv', index=False)
valid_df.to_csv('valid.csv', index=False)
test_df.to_csv('test.csv', index=False)