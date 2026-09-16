"""80/10/10 split for Track B. Filename is a typo for split.py.

`random_state=42`, *not* stratified. Realized counts in this clone:

    train 9482   valid 1185   test 1186

`model_full_SST.py` hard-codes the three output names. Re-running
this file is how you rebuild them if `combined_full_sst_et.csv`
changes. Neutral is already ~19% so the unstratified draw is close,
but a stratified rewrite would still be the cleaner default.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
file_path = 'combined_full_sst_et.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 拆分数据集为训练集和（验证集+测试集），比例为 80%:20%
train_df, valid_test_df = train_test_split(df, test_size=0.2, random_state=42)

# 将（验证集+测试集）拆分为验证集和测试集，各占 50%
valid_df, test_df = train_test_split(valid_test_df, test_size=0.5, random_state=42)

# 保存为新的 CSV 文件
train_df.to_csv('train_full_sst.csv', index=False)
valid_df.to_csv('valid_full_sst.csv', index=False)
test_df.to_csv('test_full_sst.csv', index=False)