#使用pandas加载数据
import pandas as pd
from torch_rechub.models.multi_task import MMOE
from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator
from trainers.trainer_mmoe import MTLTrainer
import sys
from sklearn.model_selection import train_test_split
from datetime import datetime

import os
# 创建 log 目录（如果不存在）
if not os.path.exists('log'):
    os.makedirs('log')

# 获取当前时间并格式化为字符串（如：2024-06-28_231141）
current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')

# 构造日志文件名，包含日期和时间
log_filename = f'/home/xyf/paper/MMoE/log/output_{current_time}.log'

# 打开日志文件
log_file = open(log_filename, 'w')

# 重定向标准输出和标准错误
sys.stdout = log_file
sys.stderr = log_file

sys.path.append('/home/xyf/paper/MMoE')
device = 'cuda' 
save_dir = '/home/xyf/paper/MMoE/saved'
learning_rate = 1e-3
epoch = 3 #10
weight_decay = 1e-5

df_total = pd.read_csv("/home/xyf/paper/MMoE/data/train_processed_target.csv")
df_train, temp_df = train_test_split(df_total, test_size=0.2, random_state=42)

# 第二次划分，将剩余集（20%）划分为验证集和测试集，各占一半
df_val, df_test = train_test_split(temp_df, test_size=0.5, random_state=42)



col_names = df_total.columns.values.tolist()
sparse_cols = ['user_id', 'net_type', 'task_id', 'cat_feature1', 'product_id', 'slot_id', 'inter_type', 'material_id', 'product_class1', 'product_class2', 'industry_id1', 'industry_id2', 'industry_id3', 'gender', 'device_price', 'province', 'city', 'age']
dense_cols = ['product_score', 'user_feature1', 'user_feature3']
used_cols = sparse_cols + dense_cols
user_cols = ['age', 'gender', 'device_price', 'province', 'city']
item_cols = [col for col in sparse_cols if col not in user_cols]
user_features = [SparseFeature(col, df_total[col].max() + 1, embed_dim=18) for col in user_cols]
item_features = [SparseFeature(col, df_total[col].max() + 1, embed_dim=18) for col in item_cols]
label_cols = ["cvr_label_1", "cvr_label_2", "cvr_label_3"]
label_target_cols = ["cvr_label_1", "cvr_label_2", "cvr_label_3", "cvr_target_1", "cvr_target_2", "cvr_target_3"]
# total_idx = len(df_total)
# train_idx = int(total_idx * 0.8)
# val_idx = train_idx + int(total_idx * 0.1)
df_train = pd.read_csv("/home/xyf/paper/MMoE/data/train_data.csv")
df_val = pd.read_csv("/home/xyf/paper/MMoE/data/val_data.csv")
df_test = pd.read_csv("/home/xyf/paper/MMoE/data/test_data.csv")


# x_train, y_train = {name: df_total[name].values[:train_idx] for name in used_cols}, df_total[label_cols].values[:train_idx]
# x_val, y_val = {name: df_total[name].values[train_idx:val_idx] for name in used_cols}, df_total[label_cols].values[train_idx:val_idx]
# x_test, y_test = {name: df_total[name].values[val_idx:] for name in used_cols}, df_total[label_cols].values[val_idx:]


x_train, y_train = {name: df_train[name].values for name in used_cols}, df_train[label_cols].values
x_val, y_val = {name: df_val[name].values for name in used_cols}, df_val[label_target_cols].values
x_test, y_test = {name: df_test[name].values for name in used_cols}, df_test[label_target_cols].values

dg = DataGenerator(x_train, y_train)
train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=x_val, y_val=y_val, 
                                      x_test=x_test, y_test=y_test, batch_size=1024)

task_types = ["classification", "classification", "classification"] 

sparse_features = [SparseFeature(col, df_total[col].max()+1, embed_dim=18) for col in sparse_cols]
dense_features = [DenseFeature(col) for col in dense_cols]
features = sparse_features + dense_features
model = MMOE(features, task_types, 8, expert_params={"dims": [32*5, 16*5, 8*5]}, tower_params_list=[{"dims": [32*5, 16*5, 8*5]}, {"dims": [32*5, 16*5, 8*5]}, {"dims": [32*5, 16*5, 8*5]}])

#训练模型及评估
mtl_trainer = MTLTrainer(model, task_types=task_types, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=30, device=device, model_path=save_dir)

mtl_trainer.fit(train_dataloader, val_dataloader)
auc = mtl_trainer.evaluate_target(mtl_trainer.model, test_dataloader)
print(auc)