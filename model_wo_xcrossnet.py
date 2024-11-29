"""
Date: create on 04/05/2022
References: 
    paper: (SIGIR'2018) Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate 
    url: https://arxiv.org/abs/1804.07931
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


from basic.features import DenseFeature, SparseFeature, SequenceFeature


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
                raise ValueError("If keep the original shape:[batch_size, num_features, embed_dim], expected %s in feature list, got %s" % ("SparseFeatures", features))
                # return dense_values


class ESMM(nn.Module):
    """Entire Space Multi-Task Model

    Args:
        user_features (list): the list of `Feature Class`, training by shared bottom and tower module. It means the user features.
        item_features (list): the list of `Feature Class`, training by shared bottom and tower module. It means the item features.
        cvr_params (dict): the params of the CVR Tower module, keys include:`{"dims":list, "activation":str, "dropout":float`}
        ctr_params (dict): the params of the CTR Tower module, keys include:`{"dims":list, "activation":str, "dropout":float`}
    """

    def __init__(self, user_features, item_features, cvr_params, ctr_params):
        super().__init__()
        # self.cross_num = 4
        # self.num_products = 20
        self.user_features = user_features
        # self.dense_features = user_features[-dense_num:]
        self.item_features = item_features
        self.embedding = EmbeddingLayer(user_features + item_features)
        # self.dense_tower_dims = len(self.dense_features) * self.dense_features[0].embed_dim
        self.tower_dims = len(self.user_features) * user_features[0].embed_dim + len(self.item_features) * item_features[0].embed_dim
        # self.tower_dims = len(self.user_features) * user_features[0].embed_dim + len(self.item_features) * item_features[0].embed_dim + (self.cross_num+1)*dense_num+2*self.num_products
        # self.dense_tower = AttentionMLP(self.dense_tower_dims, **cvr_params)
        # self.dense_tower_ctr = AttentionMLP(self.dense_tower_dims, **ctr_params)
        self.tower_cvr = MLP(self.tower_dims, **cvr_params)
        self.tower_ctr = MLP(self.tower_dims, **ctr_params)
        cl_params = {"dims": [512, 256, 128], "dropout": 0.2}
        self.tower_cl = MLP(self.tower_dims,**cl_params)
        
        
        # self.kernels = torch.nn.ParameterList(
        #     [nn.Parameter(nn.init.xavier_normal_(torch.empty(dense_num, 1)), requires_grad=True)
        #      for i in range(self.cross_num)])
        # self.bias = torch.nn.ParameterList(
        #     [nn.Parameter(nn.init.zeros_(torch.empty(dense_num, 1)), requires_grad=True)
        #      for i in range(self.cross_num)])
        # self.num_fea = 23
        # self.emb_dim = 18
        
        # self.W1 = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.num_products, self.emb_dim, self.num_fea)), requires_grad=True)
        # self.W2 = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.num_products, self.num_fea, self.num_fea, self.emb_dim)), requires_grad=True)

    def forward(self, x):
        #Field-wise Pooling Layer for user and item
        embed_user_features= self.embedding(x, self.user_features,
                                             squeeze_dim=False)  #[batch_size, num_features, embed_dim] [150, 18, 18]
        embed_item_features = self.embedding(x, self.item_features,
                                             squeeze_dim=False)  #[batch_size, num_features, embed_dim] [150, 5, 18]
        # print(self.user_features)
        # print(x)
        bsz = embed_user_features.shape[0]
        # P1 = torch.zeros(bsz, self.num_products)
        # P2 = torch.zeros(bsz, self.num_products)
        # E = torch.cat((embed_user_features, embed_item_features), dim=1)
        # for t in range(self.num_products):
        #     # 计算 P1^t
        #     P1_t = torch.einsum('bik,kj->bij', E, self.W1[t]).sum(dim=1)
        #     P1[:, t] = P1_t.sum(dim=1)
            
        #     # 计算 P2^t
        #     inner_products = torch.einsum('bik,bjk->bij', E, E)  # [bsz, num_fea, num_fea]
        #     weighted_inner_products = torch.einsum('bij,ijk->bik', inner_products, self.W2[t])  # [bsz, num_fea, emb_dim]
        #     P2_t = weighted_inner_products.sum(dim=1)
        #     P2[:, t] = P2_t.sum(dim=1)
        # P1 = P1.to('cuda:0')
        # P2 = P2.to('cuda:0')

        embed_user_features = embed_user_features.reshape(bsz, -1) #[batch_size, num_features* embed_dim] [150, 324]
        embed_item_features = embed_item_features.reshape(bsz, -1) #[batch_size, num_features* embed_dim] [150, 90]

        # embed_dense_features= self.embedding(x, self.dense_features,
        #                                      squeeze_dim=False)  #[batch_size, num_features, embed_dim] [150, 8]
        

        
        # x_0 = embed_dense_features.unsqueeze(2)
        # x_1 = x_0
        # cross_output = embed_dense_features
        # for i in range(self.cross_num):
        #     x_2 = torch.tensordot(x_1, self.kernels[i], dims=([1], [0]))
        #     x_2 = torch.matmul(x_0, x_2)
        #     x_2 = x_2 + self.bias[i]
        #     cross_output = torch.cat((x_2.squeeze(2), cross_output), dim=-1)
        #     x_1 = x_2
        


        
        
        input_tower = torch.cat((embed_user_features, embed_item_features), dim=1) # [150, 414] [150,23,18]
        # input_tower = torch.cat((input_tower, P1), dim=1)
        # input_tower = torch.cat((input_tower, P2), dim=1)
        # input_tower = torch.cat((input_tower, embed_dense_features), dim=1)
        # input_tower = torch.cat((input_tower, cross_output), dim=1)
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
        view1 = input_tower * mask1
        view2 = input_tower * mask2
        h_view1 = self.tower_cl(view1)
        h_view2 = self.tower_cl(view2)



        


        

        return torch.cat(ys, dim=1), h_view1, h_view2, input_tower
