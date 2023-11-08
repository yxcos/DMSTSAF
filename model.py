from tkinter.tix import Tree
from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import construct_st_adj


class gcn_operation(nn.Module):
    def __init__(self, adj, in_dim, out_dim, num_vertices, activation='GLU'):
        """
        图卷积模块
        :param adj: 邻接图
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param num_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(gcn_operation, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.activation = activation

        assert self.activation in {'GLU', 'relu'}

        if self.activation == 'GLU':
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def forward(self, x, mask=None):
        """
        :param x: (2*N, B, Cin)
        :param mask:(2*N, 2*N)
        :return: (2*N, B, Cout)
        """
        adj = self.adj
        if mask is not None:
            adj = adj.to(mask.device) * mask
        x = torch.einsum('nm, mbc->nbc', adj.to(x.device), x)  # 2*N, B, Cin

        if self.activation == 'GLU':
            lhs_rhs = self.FC(x)  # 2*N, B, 2*Cout
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)  # 2*N, B, Cout

            out = lhs * torch.sigmoid(rhs)
            del lhs, rhs, lhs_rhs

            return out

        elif self.activation == 'relu':
            return torch.relu(self.FC(x))  # 2*N, B, Cout


class STSGCM(nn.Module):
    def __init__(self, adj, in_dim, out_dims, num_of_vertices, activation='GLU'):
        """
        :param adj: 邻接矩阵
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(STSGCM, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation

        self.gcn_operations = nn.ModuleList()

        #self.twoNtoN=nn.Linear(2*num_of_vertices,num_of_vertices)

        self.gcn_operations.append(
            gcn_operation(
                adj=self.adj,
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],
                num_vertices=self.num_of_vertices,
                activation=self.activation
            )
        )

        for i in range(1, len(self.out_dims)):
            self.gcn_operations.append(
                gcn_operation(
                    adj=self.adj,
                    in_dim=self.out_dims[i-1],
                    out_dim=self.out_dims[i],
                    num_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )

    def forward(self, x, mask=None):
        """
        :param x: (2N, B, Cin)
        :param mask: (2N, 2N)
        :return: (N, B, Cout)
        """
        need_concat = []

        for i in range(len(self.out_dims)):
            x = self.gcn_operations[i](x, mask)
            need_concat.append(x)

        
        # 每一个都裁剪,形状从1,2N,B,C--1,N,B,C
        need_concat = [
            torch.unsqueeze(
                h[self.num_of_vertices: 2 * self.num_of_vertices], dim=0  #取N到2N
            ) for h in need_concat
        ]
        '''
        #每一个不裁剪,形状是1,2N,B,C
        need_concat = [
            torch.unsqueeze(
                h, dim=0  
            ) for h in need_concat
        ] 
        '''
        out = torch.max(torch.cat(need_concat, dim=0), dim=0).values  # (N, B, Cout)

        #out=self.twoNtoN(out.transpose(0,2)).transpose(0,2) #2N到N

        del need_concat

        return out


class STSGCL(nn.Module):
    def __init__(self,
                 adj,
                 history,
                 num_of_vertices,
                 in_dim,
                 out_dims,
                 strides_span,
                 strides=2,
                 activation='GLU',
                 temporal_emb=True,
                 spatial_emb=True):
        """
        :param adj: 两种时空图邻接矩阵[2,2N,2N]
        :param history: 输入时间步长
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为2
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        :param temporal_emb: 加入时间位置嵌入向量
        :param spatial_emb: 加入空间位置嵌入向量
        """
        super(STSGCL, self).__init__()
        self.adj = adj
        self.strides_span=strides_span
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices

        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb


        self.FC=nn.Linear(2*self.out_dims[-1],self.out_dims[-1],bias=True)
        self.conv1 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, self.strides_span))
        self.conv2 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, self.strides_span))

        self.STSGCMS0 = nn.ModuleList()
        self.STSGCMS1 = nn.ModuleList()
        for i in range(self.history - (self.strides+(self.strides-1)*(self.strides_span-1)) + 1):
            self.STSGCMS0.append(
                STSGCM(
                    adj=self.adj[0],
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation
                )            
            )
            self.STSGCMS1.append(
                STSGCM(
                    adj=self.adj[1],
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation
                )            
            )

        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.history, 1, self.in_dim))
            # 1, T, 1, Cin

        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))
            # 1, 1, N, Cin

        self.reset()

    def reset(self):
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)

        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x, mask=None):
        """
        :param x: B, T, N, Cin
        :param mask: (N, N)
        :return: B, T-2, N, Cout
        """
        if self.temporal_emb:
            x = x + self.temporal_embedding

        if self.spatial_emb:
            x = x + self.spatial_embedding

        #############################################
        # shape is (B, C, N, T)
        data_temp = x.permute(0, 3, 2, 1)
        data_left = torch.sigmoid(self.conv1(data_temp))
        data_right = torch.tanh(self.conv2(data_temp))
        data_time_axis = data_left * data_right
        data_res = data_time_axis.permute(0, 3, 2, 1)
        # shape is (B, T-2, N, C)
        #############################################
        need_concat0 = []
        need_concat1 = []
        batch_size = x.shape[0]
        for i in range(self.history - (self.strides+(self.strides-1)*(self.strides_span-1)) + 1):
            indices=[(i,i+self.strides_span)]
            t = x[:, indices, :, :]  # (B, 2, N, Cin)
            t = torch.reshape(t, shape=[batch_size, self.strides * self.num_of_vertices, self.in_dim])
            # (B, 2*N, Cin)

            t0 = self.STSGCMS0[i](t.permute(1, 0, 2), mask[0])  # (2*N, B, Cin) -> (N, B, Cout)
            t1 = self.STSGCMS1[i](t.permute(1, 0, 2), mask[1])  # (2*N, B, Cin) -> (N, B, Cout)

            t0 = torch.unsqueeze(t0.permute(1, 0, 2), dim=1)  # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)
            t1 = torch.unsqueeze(t1.permute(1, 0, 2), dim=1)  # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)

            need_concat0.append(t0)
            need_concat1.append(t1)

        out0 = torch.cat(need_concat0, dim=1)  # 四层分别为(B, 11, N, Cout)，(B, 9, N, Cout)，(B, 5, N, Cout)，(B, 1, N, Cout)
        out1 = torch.cat(need_concat1, dim=1)  # 四层分别为(B, 11, N, Cout)，(B, 9, N, Cout)，(B, 5, N, Cout)，(B, 1, N, Cout)

        z= torch.sigmoid((self.FC(torch.cat([out0,out1],dim=3))))
        out=z*out0+(1-z)*out1+data_res

        #out = out2+data_res

        del need_concat0,need_concat1, batch_size,out0,out1,z

        return out
        

class output_layer(nn.Module):
    def __init__(self, num_of_vertices, history, in_dim,
                 hidden_dim=128, horizon=12):
        """
        预测层，注意在作者的实验中是对每一个预测时间step做处理的，也即他会令horizon=1
        :param num_of_vertices:节点数
        :param history:输入时间步长
        :param in_dim: 输入维度
        :param hidden_dim:中间层维度
        :param horizon:预测时间步长
        """
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.history = history
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        self.FC1 = nn.Linear(self.in_dim * self.history, self.hidden_dim, bias=True)

        self.FC2 = nn.Linear(self.hidden_dim, self.horizon, bias=True)

    def forward(self, x):
        """
        :param x: (B, Tin, N, Cin)
        :return: (B, Tout, N)
        """
        batch_size = x.shape[0]

        x = x.permute(0, 2, 1, 3)  # B, N, Tin, Cin

        out1 = torch.relu(self.FC1(x.reshape(batch_size, self.num_of_vertices, -1)))
        # (B, N, Tin, Cin) -> (B, N, Tin * Cin) -> (B, N, hidden)

        out2 = self.FC2(out1)  # (B, N, hidden) -> (B, N, horizon)

        del out1, batch_size

        return out2.permute(0, 2, 1)  # B, horizon, N


class DMSTSN(nn.Module):
    def __init__(self, adj, history, num_of_vertices, in_dim, hidden_dims,
                 first_layer_embedding_size, out_layer_dim, activation='GLU', use_mask=True,
                 temporal_emb=True, spatial_emb=True, horizon=12, strides=2):
        """
        :param adj: local时空间矩阵
        :param history:输入时间步长
        :param num_of_vertices:节点数量
        :param in_dim:输入维度
        :param hidden_dims: lists, 中间各STSGCL层的卷积操作维度
        :param first_layer_embedding_size: 第一层输入层的维度
        :param out_layer_dim: 输出模块中间层维度
        :param activation: 激活函数 {relu, GlU}
        :param use_mask: 是否使用mask矩阵对adj进行优化
        :param temporal_emb:是否使用时间嵌入向量
        :param spatial_emb:是否使用空间嵌入向量
        :param horizon:预测时间步长
        :param strides:滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        """
        super(DMSTSN, self).__init__()
        self.adj = adj
        self.num_of_vertices = num_of_vertices
        self.hidden_dims = hidden_dims
        self.out_layer_dim = out_layer_dim
        self.activation = activation
        self.use_mask = use_mask

        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.horizon = horizon
        self.strides = strides
        
        self.strides_span=[1,2,4,4]
        #self.strides_span=1

        self.First_FC = nn.Linear(in_dim, first_layer_embedding_size, bias=True)

        self.temporalAttention=temporalAttention(64)
        self.spatialAttention=spatialAttention(64)

        self.STSGCLS = nn.ModuleList()
        self.STSGCLS.append(
            STSGCL(
                adj=self.adj,
                history=history,
                num_of_vertices=self.num_of_vertices,
                in_dim=first_layer_embedding_size,
                out_dims=self.hidden_dims[0],
                strides_span=self.strides_span[0],
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb
            )            
        )

        in_dim = self.hidden_dims[0][-1]
        history -= self.strides_span[0]
        
        #self.strides_span =2*self.strides_span

        for idx, hidden_list in enumerate(self.hidden_dims):
            if idx == 0:
                continue
            self.STSGCLS.append(
                STSGCL(
                    adj=self.adj,
                    history=history,
                    num_of_vertices=self.num_of_vertices,
                    in_dim=in_dim,
                    out_dims=hidden_list,
                    strides_span=self.strides_span[idx],
                    strides=self.strides,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb
                )
            )

            history -= self.strides_span[idx]
            in_dim = hidden_list[-1]

            #self.strides_span =2*self.strides_span

        self.predictLayer = nn.ModuleList()
        for t in range(self.horizon):
            self.predictLayer.append(
                output_layer(
                    num_of_vertices=self.num_of_vertices,
                    history=26,#这里26是四层layer的输出在T的维度拼接之后的值，所以是T1+T2+T3+T4=11+9+5+1=26,而STSGCN和STFGNN里是最后一个laye的T
                    in_dim=in_dim,
                    hidden_dim=out_layer_dim,
                    horizon=1
                )
            )

        if self.use_mask:
            mask = torch.zeros_like(self.adj)
            mask[self.adj != 0] = self.adj[self.adj != 0]
            self.mask = nn.Parameter(mask)
        else:
            self.mask = None

    def forward(self, x):
        """
        :param x: B, Tin, N, Cin)
        :return: B, Tout, N
        """

        x = torch.relu(self.First_FC(x))  # B, Tin, N, Cin
        '''
        for model in self.STSGCLS:
            x = model(x, self.mask)
        # (B, T - 8, N, Cout)
        '''
        
        #各Layer输出拼接，中间一层加注意力
        layer_output=[]
        for i in range(len(self.STSGCLS)):
            if i==len(self.STSGCLS)-2:
                x=self.spatialAttention(x)
                x=self.temporalAttention(x)
            x = self.STSGCLS[i](x, self.mask)
            layer_output.append(x)
            #layer_output.append(torch.max(x,dim=1,keepdim=True)[0]) #保存每层输出的最大池化，以备拼接
        x=torch.concat(layer_output,1) #各layer的输出拼接，变为B,T,N,C-B,T1+T2+T3+T4,N,C

        need_concat = []
        for i in range(self.horizon):
            out_step = self.predictLayer[i](x)  # (B, 1, N)
            need_concat.append(out_step)

        out = torch.cat(need_concat, dim=1)  # B, Tout, N  Tout应该是12

        del need_concat,layer_output

        return out

class spatialAttention(nn.Module):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self,D):
        super(spatialAttention, self).__init__()
        self.FC_v=nn.Linear(D,D)

    def forward(self, X):
        # [K * batch_size, num_step, num_vertex, num_vertex]
        #ASTGCN里输入一个B,T,N,C的数据集，得到的是一个B,N,N的空间注意力矩阵，没有时间维度，说明每一个时间步中都使用了相同的N,N注意力矩阵
        #GMAN这里输入一个B,T,N,C，得到的是一个B,T,N,N的空间注意力矩阵，说明在不是的时间步中使用不同的N，N注意力矩阵，更符合现实情况        
        query = X
        key = X
        value=self.FC_v(X)
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (X.shape[3] ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        del attention
        return X

class temporalAttention(nn.Module):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, D, mask=True):
        super(temporalAttention, self).__init__()
        self.D=D
        self.mask = mask
        self.FC_v=nn.Linear(D,D)

    def forward(self, X):
        # [batch_size, num_step, num_vertex, K * d]
        query = X
        key = X
        value=self.FC_v(X)
        # query: [K * batch_size, num_vertex, num_step, d]
        # key:   [K * batch_size, num_vertex, d, num_step]
        # value: [K * batch_size, num_vertex, num_step, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_step, num_step]
        attention = torch.matmul(query, key)
        attention /= (self.D ** 0.5)
        # mask attention score
        if self.mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            #attention = torch.where(mask, attention, (-2 ** 15 + 1).to(attention.device))
        # softmax
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        del query, key, value, attention
        return X