import numpy as np
from util import euclidean_distance

# 规定，如果用户不传入 N，默认 N 为 3

# 本实现默认采用欧式距离完成

# 用户也可以自定义距离函数


# 分类问题 KNN
class KNNClassifier:
    
    def __init__(self,n_neighbors = 3,distance_metric = euclidean_distance):
        '''
        :param n_neighbors: 模型使用的邻居数，默认是 3
        :param distance_metric: 可选，用户可以自定义距离函数。默认为欧式距离
                                该函数需要接收(X_train,x) 并且返回一个距离数组
        '''
        self.k = n_neighbors
        self.x_train = None
        self.y_train = None
        self.distance_metric = distance_metric


    def fit(self,x_train,y_train):
        """
        训练阶段：KNN 是懒惰学习，因此 fit 阶段只需保存数据即可
        :param X: 训练特征数据，形状为 (样本数, 特征数)
        :param y: 训练标签数据，形状为 (样本数,)
        """
        # 这里一定要转换，保证代码鲁棒性，用户传入 np.array 或者原生 list 都可以
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)


    def predict(self,x_test):
        '''
        预测阶段：对输入的多个样本进行预测
        :param x_test: 待预测的特征数据，形状为 (测试样本数, 特征数)
        :return: 预测的标签列表
        '''

        # predict_result = []
        # for x in x_test:
        #     predict_result.append(self._predict_one(x))
        # return np.array(predict_result)

        return np.array([self._predict_one(x) for x in x_test])

    def _predict_one(self,x):
        '''
        内部辅助函数：对单个样本进行预测
        :param x: 对单个训练样本预测
        '''

        # 计算传入的样本 x 与所有训练样本之间的距离
        distances = self.distance_metric(self.x_train,x)

        # 将标签值和距离按索引排序
        sorted_indics = np.argsort(distances) # 返回的结果是原索引按照键值的排序

        # 多数表决
        nearest_labels = self.y_train[sorted_indics[:self.k]]
        labels,count = np.unique(nearest_labels,return_counts=True)

        return labels[np.argmax(count)]


# 回归问题 KNN
class KNNRegression:
    def __init__(self,n_neighbors=3,distance_matrices=euclidean_distance):
        '''
        :param n_neighbors: 模型使用的邻居数，默认是 3
        :param distance_metric: 可选，用户可以自定义距离函数。默认为欧式距离
                                该函数需要接收(X_train,x) 并且返回一个距离数组
        '''
        self.k = n_neighbors
        self.distance_matrics = distance_matrices
        self.x_train = None
        self.y_train = None
    
    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self,x_test):
        '''
        预测阶段：对输入的多个样本进行预测
        :param x_test: 待预测的特征数据，形状为 (测试样本数, 特征数)
        :return: 预测的标签列表
        '''
        return np.array([self._predict_one(x) for x in x_test])

    def _predict_one(self,x):
        '''
        内部辅助函数：对单个样本进行预测
        :param x: 对单个训练样本预测
        '''

        # 计算传入的样本 x 与所有训练样本之间的距离
        distances = self.distance_matrics(self.x_train,x)

        # 将标签值和距离按索引排序
        sorted_indics = np.argsort(distances) # 返回的结果是原索引按照键值的排序

        # 进行回归流程，去前 K 个样本的平均值
        return np.mean(self.y_train[sorted_indics[:self.k]])

        
