import numpy as np

def train_test_split(X,y,test_size = 0.2,shuffle = True,random_seed = None):
    '''
    将数据集划分为训练集和测试集
    :param X: 特征矩阵，形状为 (样本数, 特征数)
    :param y: 标签数组，形状为 (样本数,)
    :param test_size: 测试集所占的比例（0 到 1 之间），默认 0.2
    :param shuffle: 布尔类型，是否打乱划分
    :param random_seed: 随机种子，固定后可确保每次运行划分的结果一致
    :return: X_train, X_test, y_train, y_test
    '''

    X = np.array(X)
    y = np.array(y)

    if test_size < 0 or test_size > 1:
        raise ValueError("test_size must be between 0 and 1.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if  X.ndim != 2:
        raise ValueError("X must be a 2D array.") # 规定：如果特征值只有一列，也要传入二维数组
    
    total_samples_number = X.shape[0]


    # 加入种子最主要的目的是为了实验可以复现
    if random_seed is not None:
        np.random.seed(random_seed)
    if shuffle:
        shuffled_indices = np.random.permutation(total_samples_number)
        X = X[shuffled_indices]
        y = y[shuffled_indices]

    test_number = int(total_samples_number * test_size)

    X_test = X[:test_number]
    y_test = y[:test_number]

    X_train = X[test_number:]
    y_train = y[test_number:]

    return X_train,X_test,y_train,y_test


