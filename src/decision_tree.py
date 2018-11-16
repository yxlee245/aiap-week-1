###### Create Decision Tree from Scratch ######

### Import libraries ###
import numpy as np

### Define functions ###
## Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

## Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

## Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

## Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

## Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)
        
## Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

## Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

## Predict single value
def predict_single(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict_single(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict_single(node['right'], row)
        else:
            return node['right']

### Define class DecicisionTree ###
class DecisionTree():
    def __init__(self, max_depth, min_size=5):
        self.max_depth, self.min_size = max_depth, min_size        
                
    def fit(self, X_train, y_train):
        # Merge X and y
        data_train = np.column_stack((X_train,y_train))
        # Convert training data to list
        self.data_train = [list(row) for row in data_train]
        # Train model on data
        self.tree = build_tree(self.data_train, self.max_depth, self.min_size)
        
    def predict(self, X_test):
        # Convert test data to list
        X_test = [list(row) for row in X_test]
        predictions = list()
        for row in X_test:
            prediction = predict_single(self.tree, row)
            predictions.append(prediction)
        return np.array(predictions)
        
    def __print__(self):
        print_tree(self.tree)


### Testing area ###
#import pandas as pd
#dataset_raw = pd.read_csv('data/titanic.csv')
#dataset_v1 = dataset_raw.copy()[['Sex', 'Fare', 'Pclass', 'Survived']]
#dataset_v1['Sex'] = (dataset_v1['Sex'] == 'male').astype(int)
#X = dataset_v1.drop(columns=['Survived']).values
#y = dataset_v1['Survived'].values
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#                                                    random_state=2018, stratify=y)
#dt = DecisionTree(5)
#dt.fit(X_train, y_train)
#dt.predict(X_test)
#from sklearn.metrics import accuracy_score
#accuracy_score(y_test, dt.predict(X_test))

## Check against sklearn tree
#from sklearn.tree import DecisionTreeClassifier
#dt_sklearn = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
#dt_sklearn.fit(X_train, y_train)
#accuracy_score(y_test, dt_sklearn.predict(X_test))