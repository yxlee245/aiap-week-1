###### Create Random Forest using DecisionTree class intances ######

### Import libraries ###
import numpy as np
import random
import pandas as pd
from src.decision_tree import DecisionTree

### Define functions ###
def bootstrapper(array, subsample_size, random_state):
    '''Creates a bootstrap sample based on inputted dataset in nested list or numpy array format,
    subsample_size which is the fraction of samples to take,
    includes a random_state number for reproducibility'''
    random.seed(random_state)
    # Create bootstrap sample
    if subsample_size > 1:
        subsample_size = 1
    bootstrap_size = int(subsample_size * len(array))
    bootstrap_sample = random.choices(array, k=bootstrap_size)
    return bootstrap_sample

### Define class RandomForest ###
class RandomForest():
    def __init__(self, max_depth=None, subsample_size=1, n_trees=5, min_size=5,
                 feature_proportion=1, max_features='sqrt', random_state=0):
        self.max_depth = max_depth
        self.subsample_size = subsample_size
        self.n_trees = n_trees
        self.min_size = min_size
        self.feature_proportion = feature_proportion
        self.max_features = max_features
        self.random_state = random_state
    
    def fit(self, X_train, y_train):
        '''Fits random forest model using training dataset
        Accepts X_train and y_train in both pandas.core.frame.DataFrame and numpy.ndarray formats'''
        # Convert X_train and y_train to arrays if they are in pandas Dataframe format
        if isinstance(X_train, pd.core.frame.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.core.frame.DataFrame):
            y_train = y_train.values
        # Fit group of trees
        trees_list = list()
        feat_ids_list = list()
        for i, tree_id in enumerate(range(self.n_trees)):
            # Sample features if feature_proportion < 1
            if self.feature_proportion < 1:
                num_features = round(self.feature_proportion * X_train.shape[1])
                if num_features < 1:
                    num_features = 1
                random.seed(self.random_state + i)
                feat_ids_array = np.random.choice(X_train.shape[1], size=num_features,
                                            replace=False)
                X_train = X_train[:, feat_ids_array]
                feat_ids_list.append(feat_ids_array)
            # Combine X_train and y_train to form data_train
            data_train = np.column_stack((X_train, y_train))
            # Create bootstrap sample
            data_train_bootstrap = bootstrapper(data_train, self.subsample_size,
                                                random_state=self.random_state + tree_id)
            data_train_bootstrap = np.array(data_train_bootstrap)
            X_train_bootstrap = data_train_bootstrap[:, :-1]
            y_train_bootstrap = data_train_bootstrap[:, -1]
            # Build decision tree on bootstrap sample
            random.seed(self.random_state + tree_id)
            tree = DecisionTree(max_depth=self.max_depth, min_size=self.min_size,
                                max_features=self.max_features)
            tree.fit(X_train_bootstrap, y_train_bootstrap)
            trees_list.append(tree)
        self.trees = trees_list
        self.feat_ids = feat_ids_list
            
    def predict(self, X_test):
        '''Predict values on test set
        Accepts X_test in both pandas.core.frame.DataFrame and numpy.ndarray formats'''
        # Convert X_test to arrays if it is in pandas Dataframe format
        if isinstance(X_test, pd.core.frame.DataFrame):
            X_test = X_test.values
        # predict results for each tree
        predictions_array = np.array([]).reshape(X_test.shape[0], 0)
        for i, tree in enumerate(self.trees):
            # Extract features corresponding to training set if feature_proportion < 1
            if self.feature_proportion < 1:
                X_test = X_test[:, self.feat_ids[i]]
            predict_list = tree.predict(X_test)
            predictions_array = np.column_stack((predictions_array, predict_list))
        # find majority vote for each row
        majority_vote_list = list()
        for row in predictions_array:
            target_unique_list = list(set(row))
            vote_result_list = [list(row).count(target) for target in target_unique_list]
            majority_vote = target_unique_list[vote_result_list.index(max(vote_result_list))]
            majority_vote_list.append(majority_vote)
        return np.array(majority_vote_list)
            
### Testing area ###
#dataset_raw = pd.read_csv('data/titanic.csv')
#dataset_v1 = dataset_raw.copy()[['Sex', 'Fare', 'Pclass', 'Survived']]
#dataset_v1['Sex'] = (dataset_v1['Sex'] == 'male').astype(int)
#X = dataset_v1.drop(columns=['Survived'])
#y = dataset_v1['Survived']
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#                                                    random_state=2018, stratify=y)
#
#rf_test = RandomForest(n_trees=3, max_depth=5, random_state=2018)
#rf_test.fit(X_train, y_train)
#rf_test.predict(X_test)
#
#from sklearn.metrics import accuracy_score
#accuracy_score(y_test, rf_test.predict(X_test))
#
#from sklearn.ensemble import RandomForestClassifier
#rf_sk = RandomForestClassifier(n_estimators=3, max_depth=5, min_samples_leaf=5,
#                               random_state=2018)
#rf_sk.fit(X_train, y_train)
#accuracy_score(y_test, rf_sk.predict(X_test))