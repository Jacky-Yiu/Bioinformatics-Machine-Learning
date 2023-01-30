from typing import Optional, Sequence, Mapping
import numpy as np
import pandas as pd
import random
pd.options.mode.chained_assignment = None

class Node(object):
    def __init__(self, node_size: int, node_class: str, depth: int, single_class:bool = False):
        # Every node is a leaf unless you set its 'children'
        self.is_leaf = True
        # Each 'decision node' has a name. It should be the feature name
        self.name = None
        # All children of a 'decision node'. Note that only decision nodes have children
        self.children = {}
        # Whether corresponding feature of this node is numerical or not. Only for decision nodes.
        self.is_numerical = None
        # Threshold value for numerical decision nodes. If the value of a specific data is greater than this threshold,
        # it falls under the 'ge' child. Other than that it goes under 'l'. Please check the implementation of
        # get_child_node for a better understanding.
        self.threshold = None
        # The class of a node. It determines the class of the data in this node. In this assignment it should be set as
        # the mode of the classes of data in this node.
        self.node_class = node_class
        # Number of data samples in this node
        self.size = node_size
        # Depth of a node
        self.depth = depth
        # Boolean variable indicating if all the data of this node belongs to only one class. This is condition that you
        # want to be aware of so you stop expanding the tree.
        self.single_class = single_class

    def set_children(self, children):
        self.is_leaf = False
        self.children = children

    def get_child_node(self, feature_value)-> 'Node':
        if not self.is_numerical:
            return self.children[feature_value]
        else:
            if feature_value >= self.threshold:
                return self.children['ge'] # ge stands for greater equal
            else:
                return self.children['l'] # l stands for less than


class RandomForest(object):
    def __init__(self, n_classifiers: int,
                 criterion: Optional['str'] = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = None,
                 max_features: Optional[int] = None):
        """
        :param n_classifiers:
            number of trees to generated in the forrest
        :param criterion:
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini
            impurity and “entropy” for the information gain.
        :param max_depth:
            The maximum depth of the trees.
        :param min_samples_split:
            The minimum number of samples required to be at a leaf node
        :param max_features:
            The number of features to consider for each tree.
        """
        self.n_classifiers = n_classifiers
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.criterion_func = self.entropy if criterion == 'entropy' else self.gini

    def fit(self, X: pd.DataFrame, y_col: str)->float:
        """
        :param X: data
        :param y_col: label column in X
        :return: accuracy of training dataset
        """
        features = self.process_features(X, y_col)
        # Your code
        for i in range(self.n_classifiers):
            print(f"generating tree {i+1}")
            bootstrap_X = X.sample(frac=1, replace=True, random_state=1)
            random_features = random.sample(features, self.max_features)
            self.trees.append(self.generate_tree(bootstrap_X, y_col, random_features))

        print(f"All tree generated, proceed to evaluation base on training set")
        return self.evaluate(X, y_col)

    def predict(self, X: pd.DataFrame)->np.ndarray:
        """
        :param X: data
        :return: aggregated predictions of all trees on X. Use voting mechanism for aggregation.
        """
        predictions = []
        # Your code
        print("collecting predictions")
        for index, row in X.iterrows():
            results_dict = dict()
            for t in self.trees:
                result = self.predict_helper(t, row)
                if result not in results_dict:
                    results_dict[result] = 1
                else:
                    results_dict[result] += 1

            predictions.append(max(results_dict, key=results_dict.get))       
        return np.array(predictions)

    def predict_helper(self, node, row)->str:
        if node.is_leaf:
            return node.node_class

        if not node.is_numerical:
            if row[node.name] not in node.children.keys():
                return "Unknown"
            else:
                return self.predict_helper(node.children[row[node.name]], row)
        else:
            if row[node.name] >= node.threshold:
                try:
                    return self.predict_helper(node.children["ge"], row)
                except KeyError:
                    return "Unknown"

            else:
                try:
                    return self.predict_helper(node.children["l"], row)
                except KeyError:
                    return "Unknown"



    def evaluate(self, X: pd.DataFrame, y_col: str)-> int:
        """
        :param X: data
        :param y_col: label column in X
        :return: accuracy of predictions on X
        """
        preds = self.predict(X)
        acc = sum(preds == X[y_col]) / len(preds)
        return acc

    def generate_tree(self, X: pd.DataFrame, y_col: str,   features: Sequence[Mapping])->Node:
        """
        Method to generate a decision tree. This method uses self.split_node() method to split a node.
        :param X:
        :param y_col:
        :param features:
        :return: root of the tree
        """
        root = Node(X.shape[0], X[y_col].mode(), 0)
        # Your code
        self.split_node(root,X,y_col,features)
        return root

    def split_node(self, node: Node, X: pd.DataFrame, y_col:str, features: Sequence[Mapping]) -> None:
        """
        This is probably the most important function you will implement. This function takes a node, uses criterion to
        find the best feature to slit it, and splits it into child nodes. I recommend to use revursive programming to
        implement this function but you are of course free to take any programming approach you want to implement it.
        :param node:
        :param X:
        :param y_col:
        :param features:
        :return:
        """
        if (node.single_class) or (node.size <= self.min_samples_split) or (node.depth >= self.max_depth) or features == None or len(features)==1:
            node.node_class = X[y_col].mode()[0]
            return
        else:
            quality= dict()
            threshold_values = dict()
            for f in features:
                if f["dtype"] != "int64":
                    quality[f["name"]] = self.criterion_func(X,f,y_col)
                else:
                    threshold_value, quality_score = self.select_threshold(X,f,y_col)
                    quality[f["name"]] = quality_score
                    threshold_values[f["name"]] = threshold_value

            if self.criterion == "gini":
                node.name = min(quality, key=quality.get)
            else:
                node.name = max(quality, key=quality.get)

            selected = None
            for f in features:
                if node.name == f["name"]:
                    selected = f
            
            if selected["dtype"] == "int64":
                node.is_numerical = True

            children = {}

            if (node.is_numerical):
                node.threshold = threshold_values[node.name]
                for i in ["l","ge"]:
                    if i == "l":
                        new_X = X[X[node.name] < node.threshold]
                    else:
                        new_X = X[X[node.name] >= node.threshold]

                    if new_X.empty != True:
                        child_node = Node(X.shape[0], X[y_col].mode(), node.depth+1)

                        if len(new_X[y_col].value_counts().keys()) == 1:
                            child_node.single_class = True

                        self.split_node(child_node, new_X, y_col, features)

                        children[i] = child_node

            else:
                for i in X[node.name].value_counts().keys():
                    new_X = X[X[node.name] == i]
                    if new_X.empty != True:
                        child_node = Node(X.shape[0], X[y_col].mode(), node.depth+1)

                        if len(new_X[y_col].value_counts().keys()) == 1:
                            child_node.single_class = True

                        self.split_node(child_node, new_X, y_col,features)

                        children[i] = child_node

            node.set_children(children)
            return

    def gini(self, X: pd.DataFrame, feature: Mapping, y_col: str) -> float:
        """
        Returns gini index of the give feature
        :param X: data
        :param feature: the feature you want to use to get compute gini score
        :param y_col: name of the label column in X
        :return:
        """
        gini = 0
        unique_value = X[feature["name"]].value_counts(normalize=True)
        for i in unique_value.keys():
            df = X[X[feature["name"]] == i]
            probs_df = df[y_col].value_counts(normalize=True)
            gini += unique_value[i] * (1-(np.power(probs_df, 2).sum()))
        return gini


    def entropy(self, X: pd.DataFrame, feature: Mapping, y_col: str) ->float:
        """
        Returns entropy of the give feature
        :param X: data
        :param feature: the feature you want to use to get compute gini score
        :param y_col: name of the label column in X
        :return:
        """
        probs_before = X[y_col].value_counts(normalize=True)
        entropy_before = -(probs_before * np.log2(probs_before)).sum()

        entropy_after = 0

        unique_value = X[feature["name"]].value_counts(normalize=True)
        for i in unique_value.keys():
          df = X[X[feature["name"]] == i]
          probs_df = df[y_col].value_counts(normalize=True)
          entropy_after += (-(probs_df * np.log2(probs_df)).sum()) * unique_value[i]

        return entropy_before - entropy_after


    def process_features(self, X: pd.DataFrame, y_col: str)->Sequence[Mapping]:
        """
        :param X: data
        :param y_col: name of the label column in X
        :return:
        """
        features = []
        for n,t in X.dtypes.items():
            if n == y_col:
                continue
            f = {'name': n, 'dtype': t}
            features.append(f)
        return features

    def select_threshold(self, X: pd.DataFrame, feature: Mapping, y_col: str):
        #unique_values = X[feature["name"]].unique()
        #unique_values.sort()
        #splits = unique_values[:-1] + np.diff(unique_values)/2
        splits=[]
        for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            splits.append(X[feature["name"]].quantile(i))
        splits = list(dict.fromkeys(splits))

        if self.criterion == "entropy":
            entro_dict={}
            probs_before = X[y_col].value_counts(normalize=True)
            entropy_before = -(probs_before * np.log2(probs_before)).sum()
            for s in splits:
                entropy_after = 0
                X["compare"] = X.apply(lambda row: 'l' if row[feature["name"]] < s  else 'ge', axis=1)
                ge_l_outcome_probs = X["compare"].value_counts(normalize=True)                 
                for i in ge_l_outcome_probs.keys():
                    df = X[X["compare"] == i]
                    probs_df = df[y_col].value_counts(normalize=True)
                    entropy_after += (-(probs_df * np.log2(probs_df)).sum()) * ge_l_outcome_probs[i]  
                entro_dict[s] = entropy_before - entropy_after

            return max(entro_dict, key=entro_dict.get) , max(entro_dict.values())

        else:
            gini_dict={}
            for s in splits:
                gini = 0
                X_copy = X.copy()
                X_copy["compare"] = X_copy.apply(lambda row: 'l' if row[feature["name"]] < s  else 'ge', axis=1)
                ge_l_outcome_probs = X_copy["compare"].value_counts(normalize=True)
                for i in ge_l_outcome_probs.keys():
                    df = X_copy[X_copy["compare"] == i]
                    probs_df = df[y_col].value_counts(normalize=True)
                    gini += ge_l_outcome_probs[i] * (1-(np.power(probs_df, 2).sum()))
                gini_dict[s] = gini

            return min(gini_dict, key=gini_dict.get) , min(gini_dict.values())

