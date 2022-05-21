import numpy as np

from decision_tree import DecisionTree
from load_data import load_titanic

def main():
    np.random.seed(123)

    train_data, test_data = load_titanic()

    dt = DecisionTree()
    dt.train(*train_data,1)
    dt.evaluate(*train_data)
    dt.evaluate(*test_data)

if __name__=="__main__":
    main()