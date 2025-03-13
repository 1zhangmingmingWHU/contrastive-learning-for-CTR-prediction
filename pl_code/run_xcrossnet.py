
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data import MyDataset
import datetime
import argparse
from model_xcrossnet import ESCM2
import pickle as pkl
from torch_rechub.basic.features import DenseFeature, SparseFeature
import pandas as pd
import numpy as np
import sys
import datetime
import os


if __name__ == "__main__":
    # 创建 log 目录（如果不存在）
    if not os.path.exists('log'):
        os.makedirs('log')

    # 获取当前时间并格式化为字符串（如：2024-06-28_231141）
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

    # 构造日志文件名，包含日期和时间
    log_filename = f'log/output_{current_time}.log'

    # 打开日志文件
    log_file = open(log_filename, 'w')

    # 重定向标准输出和标准错误
    sys.stdout = log_file
    sys.stderr = log_file


    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument("--lr", type=float, help="", default=1e-4, required=False)
    parser.add_argument("--weight_decay", type=float, help="", default=1e-5, required=False)
    parser.add_argument("--batch_size", type=int, help="", default=1024)
    parser.add_argument("--sub_batch_size", type=int, help="", default=64)
    parser.add_argument("--alpha", type=float, help="", default=0.01, required=False)
    parser.add_argument("--tau", type=float, help="", default=12, required=False)
    parser.add_argument("--devices", type=int, nargs='+', default=[0])
    parser.add_argument("--cvr_weight", type=float, nargs='+', default=0.5)
    parser.add_argument("--ctcvr_weight", type=float, nargs='+', default=0.5)
    args = parser.parse_args()
    print("======= Argument Values =======")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("===============================")



    data_path = '/data/zmm/ali-ccp' #数据存放文件夹
    df_train = pd.read_csv(data_path + '/ali_ccp_train_all.csv') #加载训练集
    df_val = pd.read_csv(data_path + '/ali_ccp_val_all.csv') #加载验证集
    df_test = pd.read_csv(data_path + '/ali_ccp_test_all.csv') #加载测试集

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%m_%d_%H_%M")



    train_idx, val_idx = df_train.shape[0], df_train.shape[0] + df_val.shape[0]
    data = pd.concat([df_train, df_val, df_test], axis=0)
    #task 1 (as cvr): main task, purchase prediction
    #task 2(as ctr): auxiliary task, click prediction
    data.rename(columns={'purchase': 'cvr_label', 'click': 'ctr_label'}, inplace=True)
    data["ctcvr_label"] = data['cvr_label'] * data['ctr_label']




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

    # # 添加dense特征
    # for col in dense_cols: 
    #     data[col] = pd.factorize(data[col])[0].astype(int)
    # dense_data = data[dense_cols]
    # unique_counts = dense_data.nunique()
    # dense_features = [DenseFeature(col,count+1,embed_dim=18) for col, count in unique_counts.items()]
    # user_features = user_features + dense_features
    dense_features = [DenseFeature(col) for col in dense_cols]
    user_features = user_features + dense_features
    used_cols = sparse_cols + dense_cols

    x_train, y_train = {name: data[name].values[:val_idx] for name in used_cols}, data[label_cols].values[:val_idx]
    x_val, y_val = {name: data[name].values[train_idx:] for name in used_cols}, data[label_cols].values[train_idx:]
    x_test, y_test = {name: data[name].values[val_idx:] for name in used_cols}, data[label_cols].values[val_idx:]


    train_dataset = MyDataset(x_train, y_train)
    val_dataset = MyDataset(x_val, y_val)
    test_dataset = MyDataset(x_test, y_test)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=64, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=64, drop_last=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=64, drop_last=False)

    task = ESCM2(lr=args.lr, weight_decay=args.weight_decay, user_features=user_features, item_features=item_features, dense_num=len(dense_cols), cvr_weight=args.cvr_weight, ctcvr_weight=args.ctcvr_weight, save_ckpt_name=formatted_time, sub_batch_size=args.sub_batch_size, tau=args.tau, alpha=args.alpha)
    trainer = pl.Trainer(accelerator='gpu', devices=args.devices, max_epochs=20, num_sanity_val_steps=0)
    trainer.fit(model=task, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
    # trainer.test(model=task, dataloaders=test_dataloader, ckpt_path="/home/xyf/paper/ESCM2_linghtning/ckpt/07_07_11_27_ctr_auc-ctr_auc=0.96110.ckpt")
    # 程序结束后，不要忘记关闭文件
    log_file.close()