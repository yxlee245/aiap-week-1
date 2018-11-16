###### Create Random Forest using DecisionTree class intances ######

### Import libraries ###
import numpy as np
import random
from src.decision_tree import DecisionTree

### Define functions ###
def bootstrapper(data_train):
    '''Creates a bootstrap sample based on inputted dataset in numpy array format'''
    

### Define class RandomForest ###
class RandomForest():
    def __init__(self, n_trees, max_depth, min_size=5):
        self.n_tress, self.max_depth, self.min_size = n_trees, max_depth, min_size



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