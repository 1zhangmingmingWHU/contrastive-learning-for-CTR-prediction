import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
from basic.callback import EarlyStopper
from utils.data import get_loss_func, get_metric_func
# from my_ESMM import ESMM
from models.multi_task import ESMM
from utils.mtl import shared_task_layers, MetaBalance
from torch_rechub.basic.features import DenseFeature, SparseFeature, SequenceFeature
import sys
from basic.layers import EmbeddingLayer
import torch.nn.functional as F



class MTLTrainer(object):
    """A trainer for multi task learning.

    Args:
        model (nn.Module): any multi task learning model.
        task_types (list): types of tasks, only support ["classfication", "regression"].
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        adaptive_params (dict): parameters of adaptive loss weight method. Now only support `{"method" : "uwl"}`. 
        n_epoch (int): epoch number of training.
        earlystop_taskid (int): task id of earlystop metrics relies between multi task (default = 0).
        earlystop_patience (int): how long to wait after last time validation auc improved (default = 10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
        alpha: 对比学习loss的系数
        tau：InfoNCE的温度
    """

    def __init__(
        self,
        user_features,
        item_features,
        model,
        task_types,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={
            "lr": 1e-3,
            "weight_decay": 1e-5
        },
        scheduler_fn=None,
        scheduler_params=None,
        adaptive_params=None,
        n_epoch=10,
        earlystop_taskid=0,
        earlystop_patience=10,
        device="cpu",
        gpus=[],
        model_path="./",
        alpha=0.01,
        tau=12,
        sub_batch_size = 64
    ):
        self.user_features = user_features
        self.item_features = item_features
        self.tau = tau
        self.alpha = alpha
        self.sub_batch_size = sub_batch_size
        self.model = model
        self.task_types = task_types
        self.n_task = len(task_types)
        self.loss_weight = None
        self.adaptive_method = None
        self.embedding = EmbeddingLayer(user_features + item_features)

        if adaptive_params is not None:
            if adaptive_params["method"] == "uwl":
                self.adaptive_method = "uwl"
                self.loss_weight = nn.ParameterList(nn.Parameter(torch.zeros(1)) for _ in range(self.n_task))
                self.model.add_module("loss weight", self.loss_weight)
            elif adaptive_params["method"] == "metabalance":
                self.adaptive_method = "metabalance"
                share_layers, task_layers = shared_task_layers(self.model)
                self.meta_optimizer = MetaBalance(share_layers)
                self.share_optimizer = optimizer_fn(share_layers, **optimizer_params)
                self.task_optimizer = optimizer_fn(task_layers, **optimizer_params)
        if self.adaptive_method != "metabalance":
            self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  #default Adam optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.loss_fns = [get_loss_func(task_type) for task_type in task_types]
        self.evaluate_fns = [get_metric_func(task_type) for task_type in task_types]
        self.n_epoch = n_epoch
        self.earlystop_taskid = earlystop_taskid
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.device = torch.device(device)

        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.model_path = model_path

    def InfoNCE_with_subbatch2(self, view1, view2, x_dict, ys, temperature, sub_batch_size, input_tower): #全部按行做log_softmax
        
        view2 = F.normalize(view2, dim=1) #[1024, 1]
        view1 = F.normalize(view1, dim=1)
        input_tower_normalized = F.normalize(input_tower, dim=1) #[1024, 414]
        ys_first_col = ys[:, 0] #[1024]
        # Compute sub_batch count
        sub_batch_count = view1.size(0) // sub_batch_size
        # Reshape views to (sub_batch_count, sub_batch_size, -1)
        view1 = view1.view(sub_batch_count, sub_batch_size, -1) #[16, 64, 1]
        view2 = view2.view(sub_batch_count, sub_batch_size, -1) #[16, 64, 1]
        input_tower_normalized = input_tower_normalized.view(sub_batch_count, sub_batch_size, -1) #[16, 64, 414]
        ys_first_col = ys_first_col.view(sub_batch_count, sub_batch_size, -1) #[16, 64, 1]
        losses = []
        for idx in range(sub_batch_count):
            sub_view1 = view1[idx] #[64, 1]
            sub_view2 = view2[idx] #[64, 1]
            sub_input_tower_normalized = input_tower_normalized[idx] #[64, 414]
            sub_ys_first_col = ys_first_col[idx]
            # 生成掩码矩阵，用于FNE
            embedding_matrix = torch.matmul(sub_input_tower_normalized, sub_input_tower_normalized.t()) #[64, 64]
            threshold = 0.9
            FNE_mask_matrix = embedding_matrix < threshold
            # 生成真实标签矩阵，用于SPI
            SPI_matrix = sub_ys_first_col & sub_ys_first_col.t() #[64, 64]
            SPI_matrix.fill_diagonal_(0)
            # 沿着 batch_size 维度拼接两个张量
            concatenated_tensor = torch.cat((sub_view1, sub_view2), dim=0)  # [sub_batch_size*2, num_features*embed_dim] [128, 1]
            # Compute similarity scores
            similarity_matrix = (concatenated_tensor @ concatenated_tensor.T) / temperature #[sub_batch_size*2, sub_batch_size*2]的方阵 [128, 128]
            # 将对角线部分设置为负无穷
            similarity_matrix.fill_diagonal_(float('-inf'))
            # 假阴性消除
            similarity_matrix = self.FNE(similarity_matrix, FNE_mask_matrix, sub_batch_size)
            # 按行做log_softmax
            log_softmax_scores = F.log_softmax(similarity_matrix, dim=1)
            # 取两个对角线
            pos_log_softmax_scores = torch.cat((torch.diagonal(log_softmax_scores, offset=sub_batch_size), torch.diagonal(log_softmax_scores, offset=-sub_batch_size)))
            # 有监督正包含
            pos_log_softmax_scores = torch.cat((pos_log_softmax_scores, self.SPI(log_softmax_scores, SPI_matrix)))
            # 计算loss
            loss = -pos_log_softmax_scores.mean()
            losses.append(loss)
        return torch.mean(torch.stack(losses))
        
    def cal_cl_loss(self, view1, view2, x_dict, ys, input_tower, temperature, sub_batch_size):
        # x_view1, x_view2 = self.item_user_encoding(x_dict)
        cl_loss = self.InfoNCE_with_subbatch2(view1 = view1, view2=view2, x_dict = x_dict, ys=ys, temperature=temperature, sub_batch_size=sub_batch_size, input_tower = input_tower)
        # cl_loss = self.InfoNCE(view1 = view1, view2=view2, temperature=self.tau)
        return cl_loss
    

    def FNE(self, similarity_matrix, mask_matrix, sub_batch_size):
        # # 转换mask_matrix为布尔类型
        # mask_matrix = mask_matrix.bool()
        
        # 直接对 similarity_matrix 的四个象限进行掩码处理
        similarity_matrix[:sub_batch_size, :sub_batch_size] = torch.where(mask_matrix, similarity_matrix[:sub_batch_size, :sub_batch_size], torch.tensor(-float('inf')))
        similarity_matrix[:sub_batch_size, sub_batch_size:] = torch.where(mask_matrix, similarity_matrix[:sub_batch_size, sub_batch_size:], torch.tensor(-float('inf')))
        similarity_matrix[sub_batch_size:, :sub_batch_size] = torch.where(mask_matrix, similarity_matrix[sub_batch_size:, :sub_batch_size], torch.tensor(-float('inf')))
        similarity_matrix[sub_batch_size:, sub_batch_size:] = torch.where(mask_matrix, similarity_matrix[sub_batch_size:, sub_batch_size:], torch.tensor(-float('inf')))

        return similarity_matrix

    
    def SPI(self, log_softmax_scores, SPI_matrix):
        # Flatten SPI_matrix to get indices where SPI_matrix is 1
        SPI_matrix_expanded = torch.cat([
        torch.cat([SPI_matrix, SPI_matrix], dim=1),
        torch.cat([SPI_matrix, SPI_matrix], dim=1)
        ], dim=0)
        indices = torch.nonzero(SPI_matrix_expanded.flatten()).squeeze()
        
        # Gather elements from log_softmax_scores using indices
        selected_elements = torch.index_select(log_softmax_scores.flatten(), 0, indices)
        
        return selected_elements


    def are_labels_equal(self, ys, label1_id, label2_id):
        if ys[label1_id][0] == ys[label2_id][0] == 1:
            return True
        else:
            return False


    def train_one_epoch(self, data_loader):
        self.model.train()
        total_loss = np.zeros(self.n_task)
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        data_len = len(data_loader)
        for iter_i, (x_dict, ys) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
            ys = ys.to(self.device)
            ctr_label = ys[:,1]
            y_preds,  view1, view2, input_tower = self.model(x_dict)
            loss_list = [self.loss_fns[i](y_preds[:, i], ys[:, i].float()) for i in range(self.n_task)]
            ctr_num = ctr_label.sum()
            PS = y_preds[:, 1] * ctr_num.float()
            PS = torch.clamp(PS, min=1e-6)
            IPS = 1.0 / PS  
            IPS = torch.clamp(IPS, min=-15, max=15)
            IPS = IPS.detach()
            batch_size = float(y_preds.size(0)) 
            IPS_scaled = IPS * batch_size
            loss_cvr_weighted = loss_list[0] * IPS_scaled  
            loss_cvr_final = (loss_cvr_weighted * ctr_label).mean()
            # loss_dense_list = [self.loss_fns[i](y_dense_preds[:, i], ys[:, i].float()) for i in range(self.n_task)]
            if isinstance(self.model, ESMM):
                loss = sum(loss_list[1:])  #ESSM only compute loss for ctr and ctcvr task
                # loss_dense = sum(loss_dense_list[1:])
            else:
                if self.adaptive_method != None:
                    if self.adaptive_method == "uwl":
                        loss = 0
                        for loss_i, w_i in zip(loss_list, self.loss_weight):
                            w_i = torch.clamp(w_i, min=0)
                            loss += 2 * loss_i * torch.exp(-w_i) + w_i
                else:
                    loss = sum(loss_list) / self.n_task
            if iter_i != data_len - 1:  # Not the last batch
                cl_loss = self.cal_cl_loss(view1, view2, x_dict, ys, input_tower, self.tau, self.sub_batch_size)  #对比学习loss
                loss = loss + self.alpha * cl_loss  #合并两个loss
                loss = loss + 0.5*loss_cvr_final
                # loss = self.alpha * cl_loss
            if self.adaptive_method == 'metabalance':
                self.share_optimizer.zero_grad()
                self.task_optimizer.zero_grad()
                self.meta_optimizer.step(loss_list)
                self.share_optimizer.step()
                self.task_optimizer.step()
            else:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            total_loss += np.array([l.item() for l in loss_list])
        log_dict = {"task_%d:" % (i): total_loss[i] / (iter_i + 1) for i in range(self.n_task)}
        print("train loss: ", log_dict)
        if self.loss_weight:
            print("loss weight: ", [w.item() for w in self.loss_weight])

    def fit(self, train_dataloader, val_dataloader):
        self.model.to(self.device)
        for epoch_i in range(self.n_epoch):
            self.train_one_epoch(train_dataloader)
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  #update lr in epoch level by scheduler
            scores = self.evaluate(self.model, val_dataloader)
            print('epoch:', epoch_i, 'validation scores: ', scores)
            if self.early_stopper.stop_training(scores[self.earlystop_taskid], self.model.state_dict()):
                print('validation best auc of main task %d: %.6f' %(self.earlystop_taskid, self.early_stopper.best_auc))
                self.model.load_state_dict(self.early_stopper.best_weights)
                torch.save(self.early_stopper.best_weights, os.path.join(self.model_path,
                                                                         "model.pth"))  #save best auc model
                break

    def evaluate(self, model, data_loader, zero_ctr_label_indices):
        self.model.to(self.device)
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, ys) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
                ys = ys.to(self.device)
                y_preds,_, _, _= self.model(x_dict)
                
                targets.extend(ys.tolist())
                predicts.extend(y_preds.tolist())
        for index in zero_ctr_label_indices:
            predicts[index][0] = 0
        targets, predicts = np.array(targets), np.array(predicts)
        scores = [self.evaluate_fns[i](targets[:, i], predicts[:, i]) for i in range(self.n_task)]
        return scores

    def predict(self, model, data_loader, zero_ctr_label_indices):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, x_dict in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y_preds = model(x_dict)
                # for index in zero_ctr_label_indices:
                #     y_preds[index] = 0
                predicts.extend(y_preds.tolist())
        return predicts






