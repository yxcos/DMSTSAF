import os
from tkinter import N
import torch
import random
import argparse
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log_string(log, string):
    """打印log"""
    log.write(string + '\n')
    log.flush()
    print(string)


def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_seed(seed):
    """Disable cudnn to maximize reproducibility 禁用cudnn以最大限度地提高再现性"""
    torch.cuda.cudnn_enabled = False
    """
    cuDNN使用非确定性算法，并且可以使用torch.backends.cudnn.enabled = False来进行禁用
    如果设置为torch.backends.cudnn.enabled =True，说明设置为使用使用非确定性算法
    然后再设置：torch.backends.cudnn.benchmark = True，当这个flag为True时，将会让程序在开始时花费一点额外时间，
    为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    但由于其是使用非确定性算法，这会让网络每次前馈结果略有差异,如果想要避免这种结果波动，可以将下面的flag设置为True
    """
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


"""图相关"""


def get_adjacency_matrix(distance_df_filename, num_of_vertices, type_='connectivity', id_filename=None):
    """
    :param distance_df_filename: str, csv边信息文件路径
    :param num_of_vertices:int, 节点数量
    :param type_:str, {connectivity, distance}
    :param id_filename:str 节点信息文件， 有的话需要构建字典
    """
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 建立映射列表
        df = pd.read_csv(distance_df_filename)
        for row in df.values:
            if len(row) != 3:
                continue
            i, j = int(row[0]), int(row[1])
            A[id_dict[i], id_dict[j]] = 1
            A[id_dict[j], id_dict[i]] = 1

        return A

    df = pd.read_csv(distance_df_filename)
    for row in df.values:
        if len(row) != 3:
            continue
        i, j, distance = int(row[0]), int(row[1]), float(row[2])
        if type_ == 'connectivity':
            A[i, j] = 1
            A[j, i] = 1
        elif type == 'distance':
            A[i, j] = 1 / distance
            A[j, i] = 1 / distance
        else:
            raise ValueError("type_ error, must be "
                             "connectivity or distance!")

    return A

def construct_st_adj(original_A, dtw_adj,pearson_adj,steps):
    #生成两个时空同步图邻接辞职，形状[2,2N,2N]
    N = original_A.shape[0]  # 获得行数
    st_adj = np.zeros((2,N * steps, N * steps)) #最终的时空图邻接矩阵
    ones_adj=np.ones((N,N)) 

    for i in range(steps):
        """两个时空图的主对角线分别用DTW图和Pearson图"""
        st_adj[0,i * N: (i + 1) * N, i * N: (i + 1) * N] = dtw_adj
        st_adj[1,i * N: (i + 1) * N, i * N: (i + 1) * N] = pearson_adj
    
    """两个时空图的副对角线分别用空间邻接图和自适应图"""
    #st_adj[0, 0: N, N: 2*N] = original_A
    st_adj[0, N: 2*N, 0: N] = original_A
    #st_adj[1, 0: N, N: 2*N] = ones_adj
    st_adj[1, N: 2*N, 0: N] = original_A
        
    #adj[N:2*N,0:N]=A #左下角是未加入自回的空间图 
    for i in range(N):
        """DTW加入自回,Pearson在生成时已经加入自回"""
        st_adj[0, i, i] = 1
    
    return torch.FloatTensor(st_adj)

def construct_pearson(dataset):
    """
    输入x_data:原始的PEMS数据集形状,PEMS04是[16992,307,3]
    返回dtw_pearson:[2,T-12-12+1,N,N],两种类型的时间图,每种类型的每个时间图形状为N*N,由12个时间步的数据计算相似性得到,一共T-23个时间步（与X,Y中的X保持一致）
    """     
    save_path='./garage/{}/pearson_adj.npy'.format(dataset) 

    if os.path.exists(save_path): 
        pearson_adj=np.load(save_path)
        print('pearson_adj loaded')
    else:
        x_path='./data/{}/{}.npz'.format(dataset,dataset) 
        data = normalize(np.load(x_path)['data'])[:, :, 0] #取第一个特征，如果输入数据只包含一个特征，这里可以不要.T,N,C--T,N
        num_nodes=data.shape[1]
        #得到N*N的皮尔逊系数矩阵
        pearson=np.corrcoef(data.T)
        pearson_adj = np.zeros([num_nodes,num_nodes])
        adj_percent = 0.01
        top = int(num_nodes * adj_percent)
        #对每个节点（即每行），取距离排名靠前的数量为top的其他节点，将邻接矩阵的对应元素赋1
        #argsort 对数组从小到大排序，并返回排序后元素在原数组中的索引
        for i in range(pearson.shape[0]):
            a = pearson[i,:].argsort()[::-1][0:top] #这里是皮尔逊相关系数，所以从大到小排序，取最大的几个
            for j in range(top):
                pearson_adj[i, a[j]] = pearson[i, a[j]] #这里也可以赋值1
                pearson_adj[a[j], i] = pearson[i, a[j]]
            #再加入自回            
            for k in range(num_nodes):
                if( i==k):
                    pearson_adj[i][k] = 1
        print("The calculation of {} pearson_adj is done!".format(dataset))  
        print(pearson_adj.shape) 
        np.save(save_path,pearson_adj)
    return pearson_adj

#STFGNN里的fast-DTW,数据输入形状为T,N,C，输入数据不能包含外部因素
def construct_dtw(x_data):
    data = x_data[:, :, 0]
    total_day = data.shape[0] / 288
    tr_day = int(total_day * 0.6)
    n_route = data.shape[1]
    xtr = gen_data(data, tr_day, n_route)
    print(np.shape(xtr))
    T0 = 288
    T = 12
    N = n_route
    d = np.zeros([N, N])
    for i in range(N):
        for j in range(i+1,N):
            d[i,j]=compute_dtw(xtr[:,:,i],xtr[:,:,j])

    print("The calculation of time series is done!")
    dtw = d+ d.T
    n = dtw.shape[0]
    w_adj = np.zeros([n,n])
    adj_percent = 0.01
    top = int(n * adj_percent)
    for i in range(dtw.shape[0]):
        a = dtw[i,:].argsort()[0:top]
        for j in range(top):
            w_adj[i, a[j]] = 1
    
    for i in range(n):
        for j in range(n):
            if (w_adj[i][j] != w_adj[j][i] and w_adj[i][j] ==0):
                w_adj[i][j] = 1
            if( i==j):
                w_adj[i][j] = 1

    print("Total route number: ", n)
    print("Sparsity of adj: ", len(w_adj.nonzero()[0])/(n*n))
    print("The weighted matrix of temporal graph is generated!")
    return w_adj

def gen_data(data, ntr, N):
    '''
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    '''
    #data=data.as_matrix()
    data=np.reshape(data,[-1,288,N])
    return data[0:ntr]

def compute_dtw(a,b,order=1,Ts=12,normal=True):    
    if normal:
        a=normalize(a)
        b=normalize(b)
    T0=a.shape[1]
    d=np.reshape(a,[-1,1,T0])-np.reshape(b,[-1,T0,1])
    d=np.linalg.norm(d,axis=0,ord=order)
    D=np.zeros([T0,T0])
    for i in range(T0):
        for j in range(max(0,i-Ts),min(T0,i+Ts+1)):
            if (i==0) and (j==0):
                D[i,j]=d[i,j]**order
                continue
            if (i==0):
                D[i,j]=d[i,j]**order+D[i,j-1]
                continue
            if (j==0):
                D[i,j]=d[i,j]**order+D[i-1,j]
                continue
            if (j==i-Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j])
                continue
            if (j==i+Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i,j-1])
                continue
            D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j],D[i,j-1])
    return D[-1,-1]**(1.0/order)

def normalize(a):
    mu=np.mean(a,axis=1,keepdims=True)
    std=np.std(a,axis=1,keepdims=True)
    return (a-mu)/std

"""数据加载器"""


class StandardScaler:
    """标准转换器"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class NScaler:
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class MinMax01Scaler:
    """最大最小值01转换器"""
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class MinMax11Scaler:
    """最大最小值11转换器"""
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        数据加载器
        :param xs:训练数据
        :param ys:标签数据
        :param batch_size:batch大小
        :param pad_with_last_sample:剩余数据不够时，是否复制最后的sample以达到batch大小
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        """洗牌"""
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


def load_dataset(dataset_dir, normalizer, batch_size, valid_batch_size=None, test_batch_size=None, column_wise=False):
    """
    加载数据集
    :param dataset_dir: 数据集目录
    :param normalizer: 归一方式
    :param batch_size: batch大小
    :param valid_batch_size: 验证集batch大小
    :param test_batch_size: 测试集batch大小
    :param column_wise: 是指列元素的级别上进行归一，否则是全样本取值
    """
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    data['x_weather']= load_weather(dataset_dir[(len(dataset_dir)-7):(len(dataset_dir)-1)],data['x_train'].shape[2]) #[B,T,N,C],其中B为X的总个数，PEMS04中是16969

    if normalizer == 'max01':
        if column_wise:
            minimum = data['x_train'].min(axis=0, keepdims=True)
            maximum = data['x_train'].max(axis=0, keepdims=True)
        else:
            minimum = data['x_train'].min()
            maximum = data['x_train'].max()

        scaler = MinMax01Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax01 Normalization')

    elif normalizer == 'max11':
        if column_wise:
            minimum = data['x_train'].min(axis=0, keepdims=True)
            maximum = data['x_train'].max(axis=0, keepdims=True)
        else:
            minimum = data['x_train'].min()
            maximum = data['x_train'].max()

        scaler = MinMax11Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax11 Normalization')

    elif normalizer == 'std':
        if column_wise:
            mean = data['x_train'].mean(axis=0, keepdims=True)  # 获得每列元素的均值、标准差
            std = data['x_train'].std(axis=0, keepdims=True)
        else:
            mean = data['x_train'].mean()
            std = data['x_train'].std()

        scaler = StandardScaler(mean, std)
        print('Normalize the dataset by Standard Normalization')

    elif normalizer == 'None':
        scaler = NScaler()
        print('Does not normalize the dataset')
    else:
        raise ValueError

    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    
    #天气数据单独进行归一化，因为交通流数据后面要反归一化才能计算误差，因此需要保存交通流的均值和方差
    #如果把交通流和天气一起归一化，那么交通流反归一化时均值和误差可能不准
    mean_weather=np.mean(data['x_weather'])
    std_weather=np.std(data['x_weather'])
    data['x_weather']=(data['x_weather']-mean_weather)/std_weather

    num_samples = data['x_weather'].shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_train - num_test

    data['x_train']=np.concatenate((data['x_train'],data['x_weather'][:num_train]),axis=3)
    data['x_val']=np.concatenate((data['x_val'],data['x_weather'][num_train:num_train+num_val]),axis=3)
    data['x_test']=np.concatenate((data['x_test'],data['x_weather'][num_train+num_val:]),axis=3)
    

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler

    return data

#生成与X形状相同的天气数据B,T,N,C，或直接读取已有文件
def load_weather(dataset,num_nodes):
    if os.path.exists('data/processed/{}/weather.npy'.format(dataset)):
        print('weather data of {} already exists'.format(dataset))
    else:
        original_weather=pd.read_excel('weather_data/processed/{}/{}.xlsx'.format(dataset,dataset))
        df_weather=pd.DataFrame(original_weather)
        np_temp=np.array(df_weather[['MaxTemp','MinTemp','Weather','Wind']])
        #将每一行的天气数据在行上重复288次，变成[day_num*288,4]
        bc_weather=np.repeat(np_temp,288,axis=0) 
        #读取训练集的样本数，将天气样本数与训练集样本数对齐 [B,T,N,C],B是切片，T,N是重复,C不变
        #train_data=np.load('data/processed/'+dataset+'/train.npz')['x']
        x_num=bc_weather.shape[0]-12-12+1
        btnc_weather=np.tile(bc_weather[0:x_num,np.newaxis,np.newaxis,:],(1,12,num_nodes,1))
        #PEMS保存的数组形状为16969,12,307,4
        np.save('data/processed/{}/weather.npy'.format(dataset),btnc_weather)
        print('generation of {} weather data is done'.format(dataset))

    return np.load('data/processed/{}/weather.npy'.format(dataset))

"""指标"""


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)

    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = torch.clip(loss,0,1)  #加了这一个裁剪的语句，某些可能是异常的label值非常小，导致单个mape非常大，影响了整体的mape，所以这里把异常的mape强制归为0到1之间

    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()

    return mae, mape, rmse

def get_distance_matrix(z):
    #z:T,N,D
    n=z.shape[1]
    distance_matrix=np.zeros((n,n))
    z=np.reshape(n,-1) #T,N,D--N,TD
    for i in range(n):
        for j in range(i,n):
            distance=distance_euclidean(z[i],z[j])
            distance_matrix[i,j]=distance
            distance_matrix[j,i]=distance
    return distance_matrix

def distance_euclidean(vector1, vector2):
    """计算欧氏距离"""
    return np.sqrt(sum(np.power(vector1-vector2, 2))) 

if __name__ == '__main__':
    adj = get_adjacency_matrix("./data/PEMS04/PEMS04.csv", 307, id_filename=None)
    print(adj)
    A = construct_adj(adj, 3)
    print(A.shape)
    print(A)

    dataloader = load_dataset('./data/processed/PEMS04/', 'std', batch_size=64, valid_batch_size=64, test_batch_size=64)
    print(dataloader)
