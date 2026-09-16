"""Dump per-subject ZuCo task-1 sentence tables from the MATLAB files.

Needs `ZuCo_mat_data/task1/*.mat` (not shipped in this clone) and
`utils_ZuCo.DataTransformer`. Writes `et_csv_data/{1-12}_SR.csv`.

Path trap: the checked-in derivatives live in `ZuCo_et_csv_data/`,
not `et_csv_data/`. Re-running this script on a machine that has the
`.mat` files will create a second folder unless you change `path`.

Scaling is *raw* here. `get_average_sentence_level.py` is the script
that builds the 12-reader mean and the min-max / z-score tables the
training run actually consumes.
"""

from utils_ZuCo import *
import os

# task1 = sentiment / normal reading (NR). raw + zeros matches the
# per-subject CSVs already in ZuCo_et_csv_data/.
datatransform_t1 = DataTransformer('task1', level='sentence', scaling='raw', fillna='zeros')

# 处理并保存每个受试者的数据
sbjs_t1 = []
for i in range(12):
    # 转换受试者数据
    sbj_data = datatransform_t1(i)

    # 将索引转换为列，并命名为 'id'
    sbj_data.reset_index(inplace=True)
    sbj_data.rename(columns={'index': 'id'}, inplace=True)

    sbjs_t1.append(sbj_data)

    # 生成文件名
    filename = f"{i+1}_SR.csv"
    # 定义保存路径
    path = os.path.join("et_csv_data", filename)

    # 显示受试者 1 的前几行数据（仅对第一个受试者执行）
    if i == 0:
        print(sbj_data.head())

    # 保存到 CSV 文件
    sbj_data.to_csv(path, index=False)  # 确保不将索引（现在是 'id' 列）再次作为索引保存

# 此处可以访问 sbjs_t1 列表中的 DataFrame