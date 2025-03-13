import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
import torch
import pickle as pkl
import numpy as np
import warnings
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
from torch_rechub.basic.features import DenseFeature, SparseFeature, SequenceFeature
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

def activation_layer(act_name):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
    
    Returns:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
        elif act_name.lower() == "softmax":
            act_layer = nn.Softmax(dim=1)
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError
    return act_layer


class ConcatPooling(nn.Module):
    """Keep the origin sequence embedding shape
   
    Shape:
    - Input: `(batch_size, seq_length, embed_dim)`
    - Output: `(batch_size, seq_length, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class AveragePooling(nn.Module):
    """Pooling the sequence embedding matrix by `mean`.
    
    Shape:
        - Input: `(batch_size, seq_length, embed_dim)`
        - Output: `(batch_size, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        sum_pooling_matrix = torch.sum(x, dim=1)
        non_padding_length = (x != 0).sum(dim=1)
        x = sum_pooling_matrix / (non_padding_length.float() + 1e-16)
        return x


class SumPooling(nn.Module):
    """Pooling the sequence embedding matrix by `sum`.

    Shape:
        - Input: `(batch_size, seq_length, embed_dim)`
        - Output: `(batch_size, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(x, dim=1)


class MLP(nn.Module):
    """Multi Layer Perceptron Module, it is the most widely used module for 
    learning feature. Note we default add `BatchNorm1d` and `Activation` 
    `Dropout` for each `Linear` Module.

    Args:
        input dim (int): input size of the first Linear Layer.
        output_layer (bool): whether this MLP module is the output layer. If `True`, then append one Linear(*,1) module. 
        dims (list): output size of Linear Layer (default=[]).
        dropout (float): probability of an element to be zeroed (default = 0.5).
        activation (str): the activation function, support `[sigmoid, relu, prelu, dice, softmax]` (default='relu').

    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)` or `(batch_size, dims[-1])`
    """

    def __init__(self, input_dim, output_layer=True, dims=[], dropout=0, activation="relu"):
        super().__init__()
        layers = list()
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(activation_layer(activation))
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    # def forward(self, x):
    #     for layer in self.mlp:
    #         x = layer(x)
    #         print(x)
    #     return x

class AttentionMLP(nn.Module):
    def __init__(self, input_dim, output_layer=True, dims=[], dropout=0, activation="relu", num_heads=1):
        super().__init__()
        self.num_features = 8
        self.embed_dim = 18
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, dropout=dropout)
        
    def forward(self, x):
        # batch_size, original_dim = x.shape
        # (batch_size, num_features * embed_dim)
        # x = x.view(-1, self.num_features, self.embed_dim)  # (batch_size, num_features, embed_dim)
        x = x.permute(1, 0, 2)  # (num_features, batch_size, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, num_features, embed_dim)
        attn_output = attn_output.reshape(attn_output.size(0), -1)  # (batch_size, num_features * embed_dim)
        # print("------------------------")
        # print(attn_output)
        # print("------------------------")
        # x = x.permute(1, 0, 2).reshape(batch_size, -1)
        #MLP
        return attn_output

class EmbeddingLayer(nn.Module):
    """General Embedding Layer. We init each embedding layer by `xavier_normal_`.
    
    Args:
        features (list): the list of `Feature Class`. It is means all the features which we want to create a embedding table.
        embed_dict (dict): the embedding dict, `{feature_name : embedding table}`.

    Shape:
        - Input: 
            x (dict): {feature_name: feature_value}, sequence feature value is a 2D tensor with shape:`(batch_size, seq_len)`,\
                      sparse/dense feature value is a 1D tensor with shape `(batch_size)`.
            features (list): the list of `Feature Class`. It is means the current features which we want to do embedding lookup.
            squeeze_dim (bool): whether to squeeze dim of output (default = `False`).
        - Output: 
            - if input Dense: `(batch_size, num_features_dense)`.
            - if input Sparse: `(batch_size, num_features, embed_dim)` or  `(batch_size, num_features * embed_dim)`.
            - if input Sequence: same with input sparse or `(batch_size, num_features_seq, seq_length, embed_dim)` when `pooling=="concat"`.
            - if input Dense and Sparse/Sequence: `(batch_size, num_features_sparse * embed_dim)`. Note we must squeeze_dim for concat dense value with sparse embedding.
    """

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.embed_dict = nn.ModuleDict()
        self.n_dense = 0

        for fea in features:
            if fea.name in self.embed_dict:  #exist
                continue
            if isinstance(fea, SparseFeature) and fea.shared_with == None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()
            elif isinstance(fea, SequenceFeature) and fea.shared_with == None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()
            elif isinstance(fea, DenseFeature):
                self.n_dense += 1

    def forward(self, x, features, squeeze_dim=False):
        sparse_emb, dense_values = [], []
        sparse_exists, dense_exists = False, False
        for fea in features:
            if isinstance(fea, SparseFeature):
                if fea.shared_with == None:
                    sparse_emb.append(self.embed_dict[fea.name](x[fea.name].long()).unsqueeze(1))
                else:
                    sparse_emb.append(self.embed_dict[fea.shared_with](x[fea.name].long()).unsqueeze(1))
            elif isinstance(fea, SequenceFeature):
                if fea.pooling == "sum":
                    pooling_layer = SumPooling()
                elif fea.pooling == "mean":
                    pooling_layer = AveragePooling()
                elif fea.pooling == "concat":
                    pooling_layer = ConcatPooling()
                else:
                    raise ValueError("Sequence pooling method supports only pooling in %s, got %s." % (["sum", "mean"], fea.pooling))
                if fea.shared_with == None:
                    sparse_emb.append(pooling_layer(self.embed_dict[fea.name](x[fea.name].long())).unsqueeze(1))
                else:
                    sparse_emb.append(pooling_layer(self.embed_dict[fea.shared_with](x[fea.name].long())).unsqueeze(1))  #shared specific sparse feature embedding
            else:
                dense_values.append(x[fea.name].float().unsqueeze(1))  #.unsqueeze(1).unsqueeze(1)


        if len(dense_values) > 0:
            dense_exists = True
            dense_values = torch.cat(dense_values, dim=1)
        if len(sparse_emb) > 0:
            sparse_exists = True
            sparse_emb = torch.cat(sparse_emb, dim=1)  #[batch_size, num_features, embed_dim]

        if squeeze_dim:  #Note: if the emb_dim of sparse features is different, we must squeeze_dim
            if dense_exists and not sparse_exists:  #only input dense features
                return dense_values
            elif not dense_exists and sparse_exists:
                return sparse_emb.flatten(start_dim=1)  #squeeze dim to : [batch_size, num_features*embed_dim]
            elif dense_exists and sparse_exists:
                return torch.cat((sparse_emb.flatten(start_dim=1), dense_values), dim=1)  #concat dense value with sparse embedding
            else:
                raise ValueError("The input features can note be empty")
        else:
            if sparse_exists:
                return sparse_emb  #[batch_size, num_features, embed_dim]
            else:
                # raise ValueError("If keep the original shape:[batch_size, num_features, embed_dim], expected %s in feature list, got %s" % ("SparseFeatures", features))
                return dense_values #[batch_size, num_features, embed_dim]


class ESMM(nn.Module):
    """Entire Space Multi-Task Model

    Args:
        user_features (list): the list of `Feature Class`, training by shared bottom and tower module. It means the user features.
        item_features (list): the list of `Feature Class`, training by shared bottom and tower module. It means the item features.
        cvr_params (dict): the params of the CVR Tower module, keys include:`{"dims":list, "activation":str, "dropout":float`}
        ctr_params (dict): the params of the CTR Tower module, keys include:`{"dims":list, "activation":str, "dropout":float`}
    """

    def __init__(self, user_features, item_features, dense_num, cvr_params, ctr_params, sum):
        super().__init__()
        self.cross_num = 4
        self.num_products = 20
        self.user_features = user_features[:-dense_num]
        self.dense_features = user_features[-dense_num:]
        self.item_features = item_features
        self.sum = sum
        self.embedding = EmbeddingLayer(user_features + item_features)
        # self.dense_tower_dims = len(self.dense_features) * self.dense_features[0].embed_dim #dense塔
        # self.dense_tower = AttentionMLP(self.dense_tower_dims, **cvr_params)
        if sum:#这个没有针对加上dense的进行修改，不要用
            self.tower_dims = user_features[0].embed_dim + item_features[0].embed_dim
        else:
            self.tower_dims = len(self.user_features) * user_features[0].embed_dim + len(self.item_features) * item_features[0].embed_dim + (self.cross_num+1)*dense_num+2*self.num_products
        
        self.tower_cvr = MLP(self.tower_dims, **cvr_params)
        self.tower_ctr = MLP(self.tower_dims, **ctr_params)
        cl_params = {"dims": [512, 256, 128], "dropout": 0.2}
        self.tower_cl = MLP(self.tower_dims,**cl_params)

        self.kernels = torch.nn.ParameterList(
            [nn.Parameter(nn.init.xavier_normal_(torch.empty(dense_num, 1)), requires_grad=True)
             for i in range(self.cross_num)])
        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(dense_num, 1)), requires_grad=True)
             for i in range(self.cross_num)])
        self.num_fea = 23
        self.emb_dim = 18
        
        self.W1 = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.num_products, self.emb_dim, self.num_fea)), requires_grad=True)
        self.W2 = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.num_products, self.num_fea, self.num_fea, self.emb_dim)), requires_grad=True)


    def forward(self, x):
        #Field-wise Pooling Layer for user and item
        embed_user_features = self.embedding(x, self.user_features,
                                             squeeze_dim=False)  #[batch_size, embed_dim]
        embed_item_features = self.embedding(x, self.item_features,
                                             squeeze_dim=False)  #[batch_size, embed_dim]
        # embed_dense_features= self.embedding(x, self.dense_features,
        #                                      squeeze_dim=False)  #[batch_size, num_features, embed_dim] [150, 18, 18]

        bsz = embed_user_features.shape[0]
        P1 = torch.zeros(bsz, self.num_products)
        P2 = torch.zeros(bsz, self.num_products)
        E = torch.cat((embed_user_features, embed_item_features), dim=1)
        for t in range(self.num_products):
            # 计算 P1^t
            P1_t = torch.einsum('bik,kj->bij', E, self.W1[t]).sum(dim=1)
            P1[:, t] = P1_t.sum(dim=1)
            
            # 计算 P2^t
            inner_products = torch.einsum('bik,bjk->bij', E, E)  # [bsz, num_fea, num_fea]
            weighted_inner_products = torch.einsum('bij,ijk->bik', inner_products, self.W2[t])  # [bsz, num_fea, emb_dim]
            P2_t = weighted_inner_products.sum(dim=1)
            P2[:, t] = P2_t.sum(dim=1)
        P1 = P1.to('cuda:0')
        P2 = P2.to('cuda:0')

        if self.sum:
            embed_user_features = torch.sum(embed_user_features, dim=1)
            embed_item_features = torch.sum(embed_item_features, dim=1)
            embed_dense_features = torch.sum(embed_dense_features, dim=1)
        else:
            embed_user_features = embed_user_features.reshape(bsz, -1)
            embed_item_features = embed_item_features.reshape(bsz, -1)
            embed_dense_features= self.embedding(x, self.dense_features,
                                             squeeze_dim=False)  #[batch_size, num_features, embed_dim] [150, 8]
        x_0 = embed_dense_features.unsqueeze(2)
        x_1 = x_0
        cross_output = embed_dense_features
        for i in range(self.cross_num):
            x_2 = torch.tensordot(x_1, self.kernels[i], dims=([1], [0]))
            x_2 = torch.matmul(x_0, x_2)
            x_2 = x_2 + self.bias[i]
            cross_output = torch.cat((x_2.squeeze(2), cross_output), dim=-1)
            x_1 = x_2

        input_tower = torch.cat((embed_user_features, embed_item_features), dim=1)
        input_tower = torch.cat((input_tower, P1), dim=1)
        input_tower = torch.cat((input_tower, P2), dim=1)
        input_tower = torch.cat((input_tower, cross_output), dim=1)
        cvr_logit = self.tower_cvr(input_tower)
        ctr_logit = self.tower_ctr(input_tower)
        cvr_pred = torch.sigmoid(cvr_logit)
        ctr_pred = torch.sigmoid(ctr_logit)
        ctcvr_pred = torch.mul(ctr_pred, cvr_pred)

        ys = [cvr_pred, ctr_pred, ctcvr_pred]

        # 生成两个不同的 mask，形状与 input_tower 相同
        mask1 = torch.rand_like(input_tower) > 0.5
        mask2 = ~mask1

        # 应用 mask，生成 view1 和 view2
        view1 = input_tower * mask1 # [1024, 414]
        view2 = input_tower * mask2 # [1024, 414]
        h_view1 = self.tower_cl(view1) # [1024, 1]
        h_view2 = self.tower_cl(view2) # [1024, 1]


        return torch.cat(ys, dim=1), h_view1, h_view2, input_tower




class ESCM2(pl.LightningModule):
    def __init__(self, lr, weight_decay, user_features, dense_num, item_features, cvr_weight, ctcvr_weight, save_ckpt_name, tau, sub_batch_size, alpha):
        super().__init__()
        self.model = ESMM(user_features, item_features, dense_num, cvr_params={"dims": [360*5, 200*5, 80*5], "dropout": 0.2}, ctr_params={"dims": [360*5, 200*5, 80*5], "dropout": 0.2}, sum=0)
        self.cvr_weight = cvr_weight
        self.ctcvr_weight = ctcvr_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_ckpt_name = save_ckpt_name
        self.tau = tau
        self.alpha = alpha
        self.sub_batch_size = sub_batch_size
        self.ctr_targets = list()
        self.ctr_predicts = list()
        self.cvr_targets = list()
        self.cvr_predicts = list()
        self.ctcvr_targets = list()
        self.ctcvr_predicts = list()


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
        SPI_matrix_expanded = torch.cat([torch.cat([SPI_matrix, SPI_matrix], dim=1), torch.cat([SPI_matrix, SPI_matrix], dim=1)], dim=0)
        indices = torch.nonzero(SPI_matrix_expanded.flatten()).squeeze()
        
        # Gather elements from log_softmax_scores using indices
        selected_elements = torch.index_select(log_softmax_scores.flatten(), 0, indices)
        
        return selected_elements

    def are_labels_equal(self, ys, label1_id, label2_id):
        if ys[label1_id][0] == ys[label2_id][0] == 1:
            return True
        else:
            return False

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

    def training_step(self, batch, batch_idx):
        loss_fct = torch.nn.BCELoss()
        loss_fct_non_reduction = torch.nn.BCELoss(reduction="none")
        label = batch.pop("label")
        cvr_label, ctr_label, ctcvr_label = label[:, 0].float(), label[:, 1].float(), label[:, 2].float()
        out, view1, view2, input_tower = self.model(batch)
        cvr_loss = loss_fct_non_reduction(out[:, 0], cvr_label)
        ctr_loss = loss_fct(out[:, 1], ctr_label)
        ctcvr_loss = loss_fct(out[:, 2], ctcvr_label)


        ctr_num = ctr_label.sum()
        PS = out[:, 1] * ctr_num.float()
        PS = torch.clamp(PS, min=1e-6)
        IPS = 1.0 / PS  
        IPS = torch.clamp(IPS, min=-15, max=15)
        IPS = IPS.detach()
        batch_size = float(out.size(0)) 
        IPS_scaled = IPS * batch_size
        loss_cvr_weighted = cvr_loss * IPS_scaled  
        loss_cvr_final = (loss_cvr_weighted * ctr_label).mean()
        # contrstive learning
        cl_loss = self.cal_cl_loss(view1, view2, batch, label, input_tower, self.tau, self.sub_batch_size)  #对比学习loss
        # cl_loss = 0



        loss = self.cvr_weight * loss_cvr_final + ctr_loss + self.ctcvr_weight * ctcvr_loss + self.alpha * cl_loss
        self.log("ctr_loss", ctr_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ctcvr_loss", ctcvr_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("cl_loss", cl_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss




    def cal_metric(self, batch):
        label = batch.pop("label")
        cvr_label, ctr_label, ctcvr_label = label[:, 0], label[:, 1], label[:, 2]
        out = self.model(batch)

        ctr_pred = (out[:, 1] >= 0.5).float() 
        ctcvr_pred = (out[:, 2] >= 0.5).float()

        ctr_auc = self.accuracy(ctr_pred, ctr_label)
        ctcvr_auc = self.accuracy(ctcvr_pred, ctcvr_label)


        return ctr_auc, ctcvr_auc
    

    def on_validation_epoch_end(self, outputs=None):
        ctr_auc = roc_auc_score(np.array(self.ctr_targets), np.array(self.ctr_predicts))
        cvr_auc = roc_auc_score(np.array(self.cvr_targets), np.array(self.cvr_predicts))
        ctcvr_auc = roc_auc_score(np.array(self.ctcvr_targets), np.array(self.ctcvr_predicts))
        self.log("ctr_auc", ctr_auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ctcvr_auc", ctcvr_auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("cvr_auc", cvr_auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.ctr_targets = list()
        self.cvr_targets = list()
        self.ctcvr_targets = list()
        self.ctr_predicts = list()
        self.cvr_predicts = list()
        self.ctcvr_predicts = list()

    
    def validation_step(self, batch, batch_idx):
        label = batch.pop("label")
        cvr_label, ctr_label, ctcvr_label = label[:, 0], label[:, 1], label[:, 2]
        out, _, _, _ = self.model(batch)
        self.cvr_predicts.extend(out[:, 0].tolist())
        self.ctr_predicts.extend(out[:, 1].tolist())
        self.ctcvr_predicts.extend(out[:, 2].tolist())
        self.cvr_targets.extend(cvr_label.tolist())
        self.ctr_targets.extend(ctr_label.tolist())
        self.ctcvr_targets.extend(ctcvr_label.tolist())



    def test_step(self, batch, batch_idx):
        label = batch.pop("label")
        cvr_label, ctr_label, ctcvr_label = label[:, 0], label[:, 1], label[:, 2]
        out = self.model(batch)
        self.cvr_predicts.extend(out[:, 0].tolist())
        self.ctr_predicts.extend(out[:, 1].tolist())
        self.ctcvr_predicts.extend(out[:, 2].tolist())
        self.cvr_targets.extend(cvr_label.tolist())
        self.ctr_targets.extend(ctr_label.tolist())
        self.ctcvr_targets.extend(ctcvr_label.tolist())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        return [optimizer]


    def configure_callbacks(self):

        checkpoint = ModelCheckpoint(dirpath="/home/zmm/HUAWEIBEI/ESCM2/ckpt",
                                    monitor="cvr_auc",
                                    filename= self.save_ckpt_name + '_cvr_auc-{cvr_auc:.5f}',
                                    save_top_k=1,
                                    mode='max',
                                    save_last=False,
                                    verbose=True,
                                    save_weights_only=False)
        return [checkpoint]
