import numpy as np
import pandas as pd
from numba import njit, prange


@njit(cache=True)
def gini(x):
    probs = (np.bincount(np.sort(x))/len(x))
    return (probs * (1 - probs)).sum()


@njit(cache=True)   
def entropy(x):
    probs = (np.bincount(np.sort(x))/len(x))
    return -(probs * np.log2(probs)).sum()


@njit
def information_gain(left_y, right_y, criterion):
    assert criterion in ["gini", "entropy"]
    criterion_fn = gini if criterion == "gini" else entropy
    left_n, right_n = len(left_y), len(right_y)
    total_n = left_n + right_n
    total_y = np.concatenate((left_y, right_y))
    result = criterion_fn(total_y) - criterion_fn(left_y)*left_n/total_n - criterion_fn(right_y)*right_n/total_n
    return result


class DecisionTreeLeaf:
    
    def __init__(self, X, y, depth, classes):
        assert (X.index == y.index).all()
        assert int(depth) == depth and depth > 0
        self.type = "leaf"
        self.X = X
        self.y = y
        self.classes = classes
        self.depth = depth
        self.volume = len(y)
        self.main_target = None
        self.target_proba = None
    
    def process_labels(self):
        probs = pd.Series(self.y).value_counts(normalize=True).sort_index()
        self.main_target = probs.argmax()
        self.target_proba = {x: probs[x] if x in probs.keys() else 0 for x in self.classes}
        
    def check_limits(self, max_depth, min_volume):
        result = True
        if max_depth and self.depth > max_depth:
            result = False
        # если текущий лист не превосходит вдвое минимальный объем - нет смысла искать его разбиение
        if self.volume < 2 * min_volume:
            result = False
        return result
            
        
class DecisionTreeNode:
    
    def __init__(self, split_dim, split_value, left, right):
        self.type = "node"
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
        

class DecisionTreeClassifier:
    
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1):
        # проверки параметров
        assert criterion in ["gini", "entropy"]
        assert max_depth is None or max_depth > 0 and int(max_depth) == max_depth
        assert min_samples_leaf > 0 and int(min_samples_leaf) == min_samples_leaf
        self.tree = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
    
    def fit(self, X, y, subspace=None):
        
        # быстрая функция для итерирования по всем возможным порогам
        @njit(cache=True)
        def iterate(X, y, min_volume, criterion):
            best_gain = 0
            best_col = -1
            best_val = 0
            for i in prange(X.shape[1]):
                X_col = X[:, i]
                values = np.sort(np.unique(X_col))
                for j in prange(1, len(values)):
                    val = values[j]
                    left_y, right_y = y[X_col < val], y[X_col >= val]
                    curr_gain = information_gain(left_y, right_y, criterion=criterion)
                    if curr_gain > best_gain and len(left_y) > min_volume and len(right_y) > min_volume:
                        best_gain = curr_gain
                        best_col = i
                        best_val = val
            return best_gain, best_col, best_val
        
        # функция для поиска лучшего разбиения
        def best_split_search(leaf):
            if not leaf.check_limits(self.max_depth, self.min_samples_leaf):
                return leaf
            X = leaf.X
            y = leaf.y
            if subspace:
                m = subspace if type(subspace) is int else subspace(X.shape[1])
                f_subset = np.random.choice(X.columns, m, replace=False)
                gain, idx, value = iterate(X[f_subset].values, y.values, self.min_samples_leaf, self.criterion)
            else:
                gain, idx, value = iterate(X.values, y.values, self.min_samples_leaf, self.criterion)
            if gain:
                dim = f_subset[idx] if subspace else X.columns[idx]
                left_X, right_X = X[X[dim] < value], X[X[dim] >= value]
                left_y, right_y = y[left_X.index], y[right_X.index]
                left = DecisionTreeLeaf(left_X, left_y, depth=leaf.depth + 1, classes=leaf.classes)
                right = DecisionTreeLeaf(right_X, right_y, depth=leaf.depth + 1, classes=leaf.classes)
                best_node = DecisionTreeNode(dim, value, left, right)
                return best_node
            else:
                return leaf
        
        # рекурсивная функция построения дерева решений
        def build_tree(leaf):
            split = best_split_search(leaf)
            if split.type == "leaf":
                split.process_labels()
                return split
            elif split.type == "node":
                left_branch = build_tree(split.left)
                right_branch = build_tree(split.right)
                split.left = left_branch
                split.right = right_branch
                return split
        
        # строим дерево от корня
        root = DecisionTreeLeaf(X, y, depth=1, classes=np.unique(y))
        self.tree = build_tree(root)
       
    
    def get_leaves_stat(self):
        leaves = []
        def find_leaves(tree):
            if tree.type == "node":
                find_leaves(tree.left)
                find_leaves(tree.right)
            else:
                leaves.append(tree)
        find_leaves(self.tree)
        stat = pd.DataFrame(index=range(len(leaves)), columns=["volume", "depth"])
        for i in range(len(leaves)):
            leaf = leaves[i]
            stat.iloc[i, :] = [leaf.X.shape[0], leaf.depth]            
        return leaves, stat
    
        
    def predict_proba(self, X):
        probs = dict.fromkeys(X.index)
        for i in X.index:
            example = X.loc[i, :]
            cursor = self.tree
            while cursor.type == "node":
                feature, value = cursor.split_dim, cursor.split_value
                if example[feature] < value:
                    cursor = cursor.left
                else:
                    cursor = cursor.right
            else:
                probs[i] = cursor.target_proba
        return pd.Series(probs)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.apply(lambda x: pd.Series(x).argmax())