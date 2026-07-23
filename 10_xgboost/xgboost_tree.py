import numpy as np


class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        predict=None,
        gain=None,
        left=None,
        right=None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.predict = predict
        self.gain = gain
        self.left = left
        self.right = right


class XGBoostTree:
    def __init__(self, max_depth=3, reg_lambda=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma

        self._root = None

    def fit(self, X, gradients, hessians):
        self._root = self._build_tree(X, gradients, hessians)
        return self

    def predict(self, X):
        return np.array([self._predict_instance(x) for x in X])

    def _predict_instance(self, x):
        curr = self._root

        while curr.feature is not None:
            if x[curr.feature] < curr.threshold:
                curr = curr.left
            else:
                curr = curr.right

        return curr.predict

    def _build_tree(self, X, gradients, hessians, depth=0):
        node = Node(predict=self._calculate_leaf_weight(gradients=gradients, hessians=hessians))

        if self.max_depth is not None and depth >= self.max_depth:
            return node

        feature, threshold, gain = self._find_best_feature_split(
            X, gradients, hessians
        )

        if feature is None:
            return node

        node.feature = feature
        node.threshold = threshold
        node.gain = gain

        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask

        node.left = self._build_tree(
            X[left_mask], gradients[left_mask], hessians[left_mask], depth + 1
        )
        node.right = self._build_tree(
            X[right_mask], gradients[right_mask], hessians[right_mask], depth + 1
        )

        return node

    def _find_best_feature_split(self, X, gradients, hessians):

        best_feature = None
        best_threshold = None
        best_gain = 0.0

        for feature in range(X.shape[1]):
            X_feature = X[:, feature]
            unique_feature = np.unique(X_feature)

            if unique_feature.size < 2:
                continue

            possible_thresholds = (unique_feature[:-1] + unique_feature[1:]) / 2

            for threshold in possible_thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask

                l_g = gradients[left_mask]
                l_h = hessians[left_mask]
                r_g = gradients[right_mask]
                r_h = hessians[right_mask]

                gain = self._get_gain(l_g, l_h, r_g, r_h)
                if gain <= 0:
                    continue
                if gain > best_gain:
                    best_feature = feature
                    best_gain = gain
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _get_gain(
        self,
        left_g,
        left_h,
        right_g,
        right_h,
    ):

        lg = np.sum(left_g)
        lh = np.sum(left_h)
        rg = np.sum(right_g)
        rh = np.sum(right_h)

        g = lg + rg
        h = lh + rh
        return (
            0.5
            * (
                lg**2 / (lh + self.reg_lambda)
                + rg**2 / (rh + self.reg_lambda)
                - g**2 / (h + self.reg_lambda)
            )
            - self.gamma
        )

    def _calculate_leaf_weight(self, gradients, hessians):
        g = np.sum(gradients)
        h = np.sum(hessians)

        if h + self.reg_lambda <= 0:
            return 0.0

        return -g / (h + self.reg_lambda)
