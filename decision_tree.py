import numpy as np
from node import Node

class DecisionTree:
    def __init__(self):
        self.root_node = Node()

    def train(self, X, y,depth):
        self.root_node.train(X, y,depth)

    def evaluate(self, X, y):
        predicted = self.predict(X)
        predicted = [round(p) for p in predicted]
        print(f"Accuracy: {round(np.mean(predicted==y),2)}")

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.root_node.predict(x))
        return prediction

