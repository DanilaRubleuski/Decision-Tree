import numpy as np
class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature_idx = None
        self.feature_value = None
        self.node_prediction = None

    def gini_best_score(self, y,possible_splits):
        best_gain = -np.inf
        best_idx = 0
        # TODO find position of best data split
        for i in possible_splits:
            zerosTop = 0
            onesTop = 0
            zerosBottom = 0
            onesBottom = 0
            for j in range(i+1):
                if y[j]==1:
                    onesTop +=1
                else:
                    zerosTop +=1
            for j in range(i+1,len(y)):
                if y[j] == 1:
                    onesBottom += 1
                else:
                    zerosBottom += 1
            pSurvivedTop = (onesTop/(onesTop+zerosTop))**2
            pSurvivedBottom = (onesBottom/(onesBottom + zerosBottom))**2
            I_t = 1 - pSurvivedTop
            I_b = 1 - pSurvivedBottom
            gain = 1-((onesTop+zerosTop)/len(y)*I_t + (onesBottom + zerosBottom)/len(y)*I_b)
            if gain >=best_gain:
                best_gain = gain
                best_idx = i
        return best_idx, best_gain

    def split_data(self, X, y, idx, val):
        left_mask = X[:, idx] < val
        return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])

    def find_possible_splits(self, data):
        possible_split_points = []
        for idx in range(data.shape[0] - 1):
            if data[idx] != data[idx + 1]:
                possible_split_points.append(idx)
        return possible_split_points

    def find_best_split(self, X, y):
        best_gain = -np.inf
        best_split = None

        for d in range(X.shape[1]):
            order = np.argsort(X[:, d])
            y_sorted = y[order]
            possible_splits = self.find_possible_splits(X[order, d])
            idx, value = self.gini_best_score(y_sorted, possible_splits)
            if value > best_gain:
                best_gain = value
                best_split = (d, [idx, idx + 1])

        if best_split is None:
            return None, None

        best_value = np.mean(X[best_split[1], best_split[0]])

        return best_split[0], best_value

    def predict(self, x):
        if self.feature_idx is None:
            return self.node_prediction
        if x[self.feature_idx] < self.feature_value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def train(self, X, y,depth):
        self.node_prediction = np.mean(y)
        if X.shape[0] == 1 or self.node_prediction == 0 or self.node_prediction == 1:
            return True

        self.feature_idx, self.feature_value = self.find_best_split(X, y)
        if self.feature_idx is None:
            return True

        (X_left, y_left), (X_right, y_right) = self.split_data(X, y, self.feature_idx, self.feature_value)

        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            self.feature_idx = None
            return True

        # create new nodes
        self.left_child, self.right_child = Node(), Node()
        if depth!=0:
            self.left_child.train(X_left, y_left,depth-1)
            self.right_child.train(X_right, y_right,depth-1)
        else:
            self.feature_idx = None
            None
