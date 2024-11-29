import pandas as pd
from tqdm import tqdm

# 定义文件路径
train_file = '/data/zmm/ali-ccp/ali_ccp_train_all.csv'
test_file = '/data/zmm/ali-ccp/ali_ccp_test_all.csv'
val_file = '/data/zmm/ali-ccp/ali_ccp_val_all.csv'

# 读取数据集
print("正在读取训练集...")
train_data = pd.read_csv(train_file)
print("正在读取测试集...")
test_data = pd.read_csv(test_file)
print("正在读取验证集...")
val_data = pd.read_csv(val_file)

# 1. 统计训练集，测试集，验证集分别有多少条数据
num_train = len(train_data)
num_test = len(test_data)
num_val = len(val_data)

print(f'训练集有 {num_train} 条数据')
print(f'测试集有 {num_test} 条数据')
print(f'验证集有 {num_val} 条数据')

# 合并三个数据集
all_data = pd.concat([train_data, test_data, val_data])

# 2. 统计click列的值为1的数据有多少条
click_count = 0
print("正在统计click列的值为1的数据...")
for value in tqdm(all_data['click'], desc="Processing clicks"):
    if value == 1:
        click_count += 1
print(f'click列的值为1的数据有 {click_count} 条')

# 3. 统计purchase列的值为1的数据有多少条
purchase_count = 0
print("正在统计purchase列的值为1的数据...")
for value in tqdm(all_data['purchase'], desc="Processing purchases"):
    if value == 1:
        purchase_count += 1
print(f'purchase列的值为1的数据有 {purchase_count} 条')