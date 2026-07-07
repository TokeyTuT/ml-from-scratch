import numpy as np
import unittest

def accuracy_score(y_true,y_pred):
    '''
    本方法用于对分类问题的预测结果进行打分
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred cannot be empty.")
    
    return (y_true == y_pred).mean()



# 测试用例类
class TestAccuracyScore(unittest.TestCase):

    def test_normal_case(self):
        """测试正常分类情况（部分预测正确）"""
        y_true = [1, 0, 1, 1, 0]
        y_pred = [1, 0, 0, 1, 1]  # 5个错2个，准确率应为 0.6
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.6)

    def test_all_correct(self):
        """测试全对的情况"""
        y_true = ['cat', 'dog', 'bird']
        y_pred = ['cat', 'dog', 'bird']
        self.assertEqual(accuracy_score(y_true, y_pred), 1.0)

    def test_all_wrong(self):
        """测试全错的情况"""
        y_true = [1, 2, 3]
        y_pred = [3, 1, 2]
        self.assertEqual(accuracy_score(y_true, y_pred), 0.0)

    def test_invalid_dimension(self):
        """测试维度不是 1D 的异常情况"""
        y_true = [[1, 2], [3, 4]]  # 2D 数组
        y_pred = [1, 2, 3, 4]
        with self.assertRaises(ValueError):
            accuracy_score(y_true, y_pred)

    def test_different_lengths(self):
        """测试两个数组长度不同的异常情况"""
        y_true = [1, 2, 3]
        y_pred = [1, 2]
        with self.assertRaises(ValueError):
            accuracy_score(y_true, y_pred)

    def test_empty_array(self):
        """测试空数组的异常情况"""
        y_true = []
        y_pred = []
        with self.assertRaises(ValueError):
            accuracy_score(y_true, y_pred)


# 运行测试
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)