from hoko.impurity import *
import numpy as np


class DecisionTextTreeNode:
    def __init__(self, kwords_list, type="bag", depth=-1, eps=1e-7):
        self.kwords_list = set(kwords_list)
        self.type = type
        self.depth = depth
        self.eps = eps

        if self.type == "bag":
            self.y_values = None
            self.y_probs = None
        elif self.type == "stump":
            self.kword = None

    def fit(self, X_text, Y):
        Y = np.array(Y)
        if self.type == "stump":
            if gini_impurity(np.unique(Y, return_counts=True)[1]) < self.eps:
                self.type = "bag"
            else:
                min_gini = 1.0
                self.kword = None

                for kword in self.kwords_list:
                    X_map = np.array(list(map(lambda x: kword in x, X_text)))

                    if not (X_map.any() and (~X_map).any()):
                        continue

                    current_gini = gini_split(
                        np.unique(Y[X_map], return_counts=True)[1],
                        np.unique(Y[~X_map], return_counts=True)[1],
                    )

                    if current_gini < min_gini:
                        min_gini = current_gini
                        self.kword = kword

            if self.kword is None:
                self.type = "bag"

        if self.type == "bag":
            self.y_values, self.y_probs = np.unique(Y, return_counts=True)
            self.y_probs = self.y_probs / np.sum(self.y_probs)

    def __repr__(self):
        if self.type == "bag":
            return f"Bag [\n  {self.y_values},\n  {self.y_probs}\n]"
        elif self.type == "stump":
            return f"Stump [\n  {self.kword=}\n]"

    def __str__(self):
        return self.__repr__()

    def pprint(self, tabs):
        if self.type == "bag":
            return f"{tabs}Bag [\n{tabs}  {self.y_values},\n{tabs}  {self.y_probs}\n{tabs}]"
        elif self.type == "stump":
            return f"{tabs}Stump [\n{tabs}  {self.kword=}\n{tabs}]"

    def predict(self, X_text):
        if self.type == "bag":
            # return np.random.choice(
            #     a=self.y_values, p=self.y_probs, size=X_text.shape[0]
            # )
            return np.full(shape=X_text.shape[0], fill_value=self.y_values[np.argmax(self.y_probs)])
        elif self.type == "stump":
            return None

    def map_predict(self, X_text):
        if self.type == "stump":
            return np.array(list(map(lambda x: self.kword in x, X_text)))
        elif self.type == "bag":
            return None


class DecisionTextTree:
    def __init__(self, kwords_list, depth=-1, eps=1e-7):
        self.kwords_list = kwords_list
        self.depth = depth
        self.eps = eps
        self.left = None
        self.right = None
        self.node = None

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        if self.depth == -1 or self.depth > 0:
            self.node = DecisionTextTreeNode(kwords_list=self.kwords_list, type="stump")
        else:
            self.node = DecisionTextTreeNode(kwords_list=self.kwords_list, type="bag")
        self.node.fit(X, Y)

        if self.node.type == "stump":
            X_map = self.node.map_predict(X)

            X_left = X[X_map]
            Y_left = Y[X_map]
            self.left = DecisionTextTree(
                kwords_list=self.kwords_list,
                depth=self.depth if self.depth == -1 else self.depth - 1,
                eps=self.eps,
            )
            self.left.fit(X_left, Y_left)

            X_right = X[~X_map]
            Y_right = Y[~X_map]
            self.right = DecisionTextTree(
                kwords_list=self.kwords_list,
                depth=self.depth if self.depth == -1 else self.depth - 1,
                eps=self.eps,
            )
            self.right.fit(X_right, Y_right)

    def pprint(self, append="  "):
        queue = [
            (self, ""),
        ]

        while len(queue) > 0:
            this, tabs = queue.pop()
            if this is None:
                continue

            print(this.node.pprint(tabs))
            queue.append((this.left, tabs + append))
            queue.append((this.right, tabs + append))

    def predict(self, X_text):
        if self.node.type == "bag":
            return self.node.predict(X_text)
        elif self.node.type == "stump":
            X_text = np.array(X_text)
            X_map = self.node.map_predict(X_text)
            Y = np.empty_like(X_text)

            X_left = X_text[X_map]
            Y[X_map] = self.left.predict(X_left)

            X_right = X_text[~X_map]
            Y[~X_map] = self.right.predict(X_right)

            return Y
