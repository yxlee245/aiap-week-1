## AIAP Week #1

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Data
dataset = pd.read_csv('data/titanic.csv')

## Initial Model

# Clean Data by filling NaNs with dummy values
dataset_nonans = dataset.copy()
dataset_nonans['Age'].fillna(999, inplace=True)
dataset_nonans['Cabin'].fillna('NA', inplace=True)
dataset_nonans['Embarked'].fillna('NA', inplace=True)
X = dataset_nonans.drop('Survived', axis=1)
y = dataset_nonans['Survived']

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
label_encoder_name = LabelEncoder()
X['Name'] = label_encoder_name.fit_transform(X['Name'])
labelencoder_sex = LabelEncoder()
X['Sex'] = labelencoder_sex.fit_transform(X['Sex'])
labelencoder_ticket = LabelEncoder()
X['Ticket'] = labelencoder_ticket.fit_transform(X['Ticket'])
labelencoder_cabin = LabelEncoder()
X['Cabin'] = labelencoder_cabin.fit_transform(X['Cabin'])
labelencoder_embark = LabelEncoder()
X['Embarked'] = labelencoder_embark.fit_transform(X['Embarked'])
X_dummies = pd.get_dummies(X, columns=['Embarked'], drop_first=True)

# Split data into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=2018,
                                                    stratify=y)

# Fit random forest without paramter tuning
from sklearn.ensemble import RandomForestClassifier
rf_initial = RandomForestClassifier()
rf_initial.fit(X_train, y_train)

# Determine accuracy using test set
from sklearn.metrics import accuracy_score
accuracy_initial = accuracy_score(y_test, rf_initial.predict(X_test))
print('Accuracy of initial model: {:.1f}'.format(accuracy_initial))

## Exploring the Data

# Split into features and target
X = dataset.drop(['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
y = dataset['Survived']

# Checking for missing data
X.info()

# Correlation matrix and plot
correlations = X.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, 11, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(X.columns)
ax.set_yticklabels(X.columns)
plt.show()

# Variable: Survived
# Show percentages of Survived
print('Percentages:')
print(y.value_counts(normalize=True)*100)

# Variable: Pclass
# Show percentages of Pclass
print('Percentages:')
print(X['Pclass'].value_counts(normalize=True)*100)

# Variables: Sex
# Show percentages of Sex
print('Percentages:')
print(X['Sex'].value_counts(normalize=True)*100)

# Variable: Age
# Histogram of Age
X['Age'].dropna(inplace=False).hist()
plt.vlines(np.mean(X['Age'].dropna(inplace=False)), ymin=0, ymax=200, color='red')
plt.vlines(np.median(X['Age'].dropna(inplace=False)), ymin=0, ymax=200, color='green')

# Boxplot of Age
dataset.boxplot(column='Age')
plt.xlabel('')

# Boxplot of Age by Survived
dataset.boxplot(column='Age', by='Survived')
plt.title('')
plt.ylabel('Age')