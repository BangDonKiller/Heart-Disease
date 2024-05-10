import numpy as np

# 定義節點類別
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # 特徵索引
        self.threshold = threshold  # 分割閾值
        self.left = left  # 左子樹
        self.right = right  # 右子樹
        self.value = value  # 預測值（葉子節點才有）

# 定義Decision Tree類別
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    # 分割資料
    def split_data(self, X, y, feature_index, threshold):
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]
        left_X, left_y = X[left_indices], y[left_indices]
        right_X, right_y = X[right_indices], y[right_indices]
        return left_X, left_y, right_X, right_y

    # 計算Gini不純度
    def gini(self, y):
        classes = np.unique(y)
        gini = 0
        total_samples = len(y)
        for c in classes:
            p = np.sum(y == c) / total_samples
            gini += p * (1 - p)
        return gini

    # 找到最佳分割特徵及閾值
    def find_best_split(self, X, y):
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_X, left_y, right_X, right_y = self.split_data(X, y, feature_index, threshold)
                gini_left = self.gini(left_y)
                gini_right = self.gini(right_y)
                gini_index = (len(left_y) / len(y)) * gini_left + (len(right_y) / len(y)) * gini_right
                if gini_index < best_gini:
                    best_gini = gini_index
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    # 建立Decision Tree
    def build_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return Node(value=np.argmax(np.bincount(y)))

        best_feature_index, best_threshold = self.find_best_split(X, y)
        if best_feature_index is None:
            return Node(value=np.argmax(np.bincount(y)))

        left_X, left_y, right_X, right_y = self.split_data(X, y, best_feature_index, best_threshold)
        left_child = self.build_tree(left_X, left_y, depth + 1)
        right_child = self.build_tree(right_X, right_y, depth + 1)

        return Node(feature_index=best_feature_index, threshold=best_threshold, left=left_child, right=right_child)

    # 預測單一樣本
    def predict_sample(self, tree, sample):
        if tree.value is not None:
            return tree.value
        if sample[tree.feature_index] <= tree.threshold:
            return self.predict_sample(tree.left, sample)
        else:
            return self.predict_sample(tree.right, sample)

    # 預測資料集
    def predict(self, tree, X):
        predictions = []
        for sample in X:
            predictions.append(self.predict_sample(tree, sample))
        return np.array(predictions)

# 載入資料集
import pandas as pd
data = pd.read_csv('./dataset/heart.csv')
X = data.drop('target', axis=1).values
y = data['target'].values

# 分割資料集為訓練集和測試集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立Decision Tree模型並進行訓練
tree = DecisionTree(max_depth=8)
tree_model = tree.build_tree(X_train, y_train)

print(X_test)

# 預測測試集
predictions = tree.predict(tree_model, X_test)

print("Predictions:", predictions)
print("True:", y_test)

# 計算準確率
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)
