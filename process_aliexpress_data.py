import pandas as pd
from tqdm import tqdm  # 导入 tqdm

# 读取用户特征文件（无列名）
user_df = pd.read_csv('/data/zmm/aliexpress_US_datasets/us_user_test/us_user_test.csv', header=None)

# 为用户特征文件生成列名
user_columns = ['pv_id'] + [f'user_feature_{i}' for i in range(1, 33)]  # 32个用户特征
user_df.columns = user_columns

# 读取项目特征文件（无列名）
item_df = pd.read_csv('/data/zmm/aliexpress_US_datasets/us_item_test/us_item_test.csv', header=None)

# 为项目特征文件生成列名
item_columns = ['pv_id'] + [f'item_feature_{i}' for i in range(1, 48)] + ['label']  # 47个项目特征 + label
item_df.columns = item_columns

# 合并两个数据集，基于 pv_id 列
merged_df = pd.merge(user_df, item_df, on='pv_id', how='inner')

# 处理 label 字段
def transform_label(row):
    if row['label'] == 0:
        return 0, 0  # impression
    elif row['label'] == 1:
        return 1, 0  # click
    elif row['label'] == 2:
        return 1, 1  # purchase
    else:
        return 0, 0  # 默认值

# 使用 tqdm 包装 apply 函数来显示进度条
tqdm.pandas(desc="Processing rows")  # 设置进度条描述
merged_df[['click', 'purchase']] = merged_df.progress_apply(lambda row: pd.Series(transform_label(row)), axis=1)

# 删除原始的 label 列
merged_df.drop(columns=['label'], inplace=True)

# 保存最终的 CSV 文件，包含列名
merged_df.to_csv('/data/zmm/aliexpress_US_datasets/US_test.csv', index=False)

print("数据处理完成，结果已保存到 /data/zmm/aliexpress_US_datasets/US_test.csv")
