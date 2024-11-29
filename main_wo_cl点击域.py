#使用pandas加载数据
import pandas as pd
# from torch_rechub.models.multi_task import ESMM
from model_wo_cl import ESMM
from basic.features import DenseFeature, SparseFeature
import torch
import os
from trainers.trainer_wo_cl import MTLTrainer 
from utils.data import DataGenerator
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm

import sys
from datetime import datetime
import os
# 创建 log 目录（如果不存在）
if not os.path.exists('log'):
    os.makedirs('log')

# 获取当前时间并格式化为字符串（如：2024-06-28_231141）
current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')

# 构造日志文件名，包含日期和时间
log_filename = f'log/output_{current_time}.log'

# 打开日志文件
log_file = open(log_filename, 'w')

# 重定向标准输出和标准错误
sys.stdout = log_file
sys.stderr = log_file

parser = argparse.ArgumentParser(description='args')
parser.add_argument("--lr", type=float, help="", default=1e-4, required=False)
parser.add_argument("--weight_decay", type=float, help="", default=1e-5, required=False)
parser.add_argument("--batch_size", type=int, help="", default=1024)
parser.add_argument("--sub_batch_size", type=int, help="", default=64)
parser.add_argument("--alpha", type=float, help="", default=0.01, required=False)
parser.add_argument("--tau", type=float, help="", default=12, required=False)
parser.add_argument("--earlystop_patience", type=int, help="", default=20)
args = parser.parse_args()

print("======= Argument Values =======")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
print("===============================")



data_path = '/data/zmm/ali-ccp' #数据存放文件夹
df_train = pd.read_csv(data_path + '/ali_ccp_train_all.csv') #加载训练集
df_val = pd.read_csv(data_path + '/ali_ccp_val_all.csv') #加载验证集
df_test = pd.read_csv(data_path + '/ali_ccp_test_all.csv') #加载测试集
# print("train : val : test = %d %d %d" % (len(df_train), len(df_val), len(df_test)))
#查看数据，其中'click'、'purchase'为标签列，'D'开头为dense特征列，其余为sparse特征，各特征列的含义参考官网描述
# print(df_train.head(5)) 

current_time = datetime.now()
formatted_time = current_time.strftime("%m_%d_%H_%M")


train_idx, val_idx = df_train.shape[0], df_train.shape[0] + df_val.shape[0]
data = pd.concat([df_train, df_val, df_test], axis=0)
#task 1 (as cvr): main task, purchase prediction
#task 2(as ctr): auxiliary task, click prediction
data.rename(columns={'purchase': 'cvr_label', 'click': 'ctr_label'}, inplace=True)
data["ctcvr_label"] = data['cvr_label'] * data['ctr_label']

zero_ctr_label_indices = df_test[df_test['ctr_label'] == 0].index#找到所有ctr为0的索引


col_names = data.columns.values.tolist()
dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
sparse_cols = [col for col in col_names if col not in dense_cols and col not in ['cvr_label', 'ctr_label', 'ctcvr_label']]
print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))
label_cols = ['cvr_label', 'ctr_label', "ctcvr_label"]  #the order of 3 labels must fixed as this
used_cols = sparse_cols #ESMM only for sparse features in origin paper  
item_cols = ['205', '206', '207', '210', '216']  #assumption features split for user and item
user_cols = [col for col in used_cols if col not in item_cols]
user_features = [SparseFeature(col, data[col].max() + 1, embed_dim=18) for col in user_cols]
item_features = [SparseFeature(col, data[col].max() + 1, embed_dim=18) for col in item_cols]

# for col in dense_cols:
#     data[col] = pd.factorize(data[col])[0].astype(int)
# dense_data = data[dense_cols]
# unique_counts = dense_data.nunique()
# dense_features = [DenseFeature(col,count+1,embed_dim=18) for col, count in unique_counts.items()]
# user_features = user_features + dense_features
dense_features = [DenseFeature(col) for col in dense_cols]
user_features = user_features + dense_features
used_cols = sparse_cols + dense_cols

model = ESMM(user_features, item_features, dense_num=len(dense_cols), cvr_params={"dims": [360*10, 200*10, 80*10], "dropout": 0.2}, ctr_params={"dims": [360*10, 200*10, 80*10], "dropout": 0.2})
model.load_state_dict(torch.load('path_to_your_model.pth'))


x_train, y_train = {name: data[name].values[:train_idx] for name in used_cols}, data[label_cols].values[:train_idx]
x_val, y_val = {name: data[name].values[train_idx:val_idx] for name in used_cols}, data[label_cols].values[train_idx:val_idx]
x_test, y_test = {name: data[name].values[val_idx:] for name in used_cols}, data[label_cols].values[val_idx:]

import numpy as np
# 合并 x_train 和 x_val，并存储到 x_new_train
x_new_train = {name: np.concatenate((x_train[name], x_val[name]), axis=0) for name in used_cols}

# 合并 y_train 和 y_val，并存储到 y_new_train
y_new_train = np.concatenate((y_train, y_val), axis=0)

# # 筛选 ctr_label 为 1 的测试集数据
# cvr_label_filter = data['ctr_label'].values[val_idx:] == 1
# x_new_test = {name: x_test[name][cvr_label_filter] for name in used_cols}
# y_new_test = y_test[cvr_label_filter]

# # 保留一条 ctr_label 为 0 的数据
# one_cvr_label_idx = np.where(data['ctr_label'].values[val_idx:] == 0)[0][0]
# x_one_cvr_label = {name: x_test[name][one_cvr_label_idx:one_cvr_label_idx+1] for name in used_cols}
# y_one_cvr_label = y_test[one_cvr_label_idx:one_cvr_label_idx+1]

# # 将保留的数据添加到 x_new_test 和 y_new_test 中
# x_new_test = {name: np.concatenate((x_new_test[name], x_one_cvr_label[name]), axis=0) for name in used_cols}
# y_new_test = np.concatenate((y_new_test, y_one_cvr_label), axis=0)

dg = DataGenerator(x_train, y_train)
train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=x_val, y_val=y_val, 
                                      x_test=x_test, y_test=y_test, batch_size=args.batch_size)



# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda:0'

epoch = 20 #10
save_dir = f'/home/zmm/HUAWEIBEI/code/saved_models/{formatted_time}'


if not os.path.exists(save_dir):
    os.makedirs(save_dir)
task_types = ["classification", "classification", "classification"] #CTR与CVR均为二分类任务
mtl_trainer = MTLTrainer(model = model, task_types=task_types, 
              optimizer_params={"lr": args.lr, "weight_decay": args.weight_decay}, 
              n_epoch=epoch, earlystop_patience=args.earlystop_patience, device=device, model_path=save_dir)
# mtl_trainer = MTLTrainer(model=model, user_features=user_features, item_features=item_features, task_types=task_types, #用加了对比学习的trainer
#               optimizer_params={"lr": args.lr, "weight_decay": args.weight_decay}, 
#               n_epoch=epoch, earlystop_patience=args.earlystop_patience, device=device, model_path=save_dir, 
#               alpha = args.alpha, tau = args.tau, sub_batch_size=args.sub_batch_size)
mtl_trainer.fit(train_dataloader, val_dataloader)
# mtl_trainer.fit(train_dataloader, test_dataloader) #用测试集作为验证集
auc = mtl_trainer.evaluate(mtl_trainer.model, test_dataloader)
print("auc: ", auc)

# 程序结束后，不要忘记关闭文件
log_file.close()