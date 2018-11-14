## AIAP Week #1

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Data
dataset_raw = pd.read_csv('data/titanic.csv')

## Initial Model

# Clean Data by filling NaNs with dummy values
dataset_nonans = dataset_raw.copy()
dataset_nonans['Age'].fillna(999, inplace=True)
dataset_nonans['Cabin'].fillna('NA', inplace=True)
dataset_nonans['Embarked'].fillna('NA', inplace=True)
X = dataset_nonans.drop(['PassengerId', 'Survived'], axis=1)
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

# Create new copy of dataset
dataset_v2 = dataset_raw.copy()
dataset_v2 = dataset_v2.drop(['PassengerId'], axis=1)

# Checking for missing data
dataset_v2.info()

# Correlation matrix and plot
correlations = dataset_v2.drop('Survived', axis=1).corr()
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
print(dataset_v2['Survived'].value_counts(normalize=True)*100)

# Variable: Pclass
# Show percentages of Pclass
print('Percentages:')
print(dataset_v2['Pclass'].value_counts(normalize=True)*100)

# Crosstab plot of Pclass
pclass_crosstab = pd.crosstab(index=dataset_v2['Pclass'],
                                columns=dataset_v2['Survived'])
pclass_crosstab.plot(kind='bar',
                       stacked=False)

# Variables: Sex
# Show percentages of Sex
print('Percentages:')
print(dataset_v2['Sex'].value_counts(normalize=True)*100)

# Crosstab plot of Sex
sex_crosstab = pd.crosstab(index=dataset_v2['Sex'],
                           columns=dataset_v2['Survived'])
sex_crosstab.plot(kind='bar',
                       stacked=False)

# Variable: Age
# Histogram of Age
dataset_v2['Age'].dropna(inplace=False).hist()
#plt.vlines(np.mean(dataset_v2['Age'].dropna(inplace=False)), ymin=0, ymax=200, color='red')
#plt.vlines(np.median(dataset_v2['Age'].dropna(inplace=False)), ymin=0, ymax=200, color='green')

# Boxplot of Age
#dataset_v2.boxplot(column='Age')
#plt.xlabel('')

# Boxplot of Age by Survived
dataset_v2.boxplot(column='Age', by='Survived')
plt.title('')
plt.ylabel('Age')

# Define Function to Extract Salutation from 'Name' column
# 'Name' strings are in FamilyName, Salutation. FirstName.... format
def salutation_extract(input_str):
    '''Parses name string and return salutation'''
    # Remove spaces in name string
    input_str.replace(' ', '')
    # Split input_str by comma
    substr_1_list = input_str.split(',')
    # Split substr_1 by period
    substr_2_list = substr_1_list[1].split('.')
    return substr_2_list[0].strip()

# Define wrapper function to create list of salutations for each passenger
def salutation_list_create(name_list):
    '''Takes in list/arrays/series of names and
    returns corresponding list of salutations'''
    # Initialize salutation_list
    salutation_list = []
    # For loop to extract salutations in name list
    for name in name_list:
        salutation = salutation_extract(name)
        salutation_list.append(salutation)
    return salutation_list

# Create list of salutations and add to dataset_v2
salutation_list = salutation_list_create(dataset_v2['Name'])
dataset_v2['Salutation'] = salutation_list
dataset_v2['Salutation'].value_counts()

# Filter out observations by Salutation for next box plot
# Only Mr, Mrs, Miss and Master chosen due to significant number of
# observations in these classes
mask = [i for i, salutation in enumerate(dataset_v2['Salutation'])\
        if salutation in ['Mr', 'Miss', 'Mrs', 'Master']]
dataset_v2_filtered = dataset_v2.iloc[mask, :]

# Boxplot of Age by Salutation
dataset_v2_filtered.boxplot(column='Age', by='Salutation')
plt.title('')
plt.ylabel('Age')

# Create Variable Num_Relative
dataset_v2['Num_Relative'] = dataset_v2['SibSp'] + dataset_v2['Parch']

# Histogram of Num_Relative
dataset_v2['Num_Relative'].hist()

# Boxplot of Num_Relative by Survived
dataset_v2.boxplot(column='Num_Relative', by='Survived')
plt.title('')
plt.ylabel('Num_Relative')

# Create Variable Ticket_Count
# Dataframe of ticket numbers with counts
ticket_num_counts = pd.DataFrame(dataset_v2['Ticket'].drop_duplicates())

# Obtain counts of every unique ticket number and add to ticket_num_counts
# dataframe
ticket_count_list = []
for ticket_num in ticket_num_counts['Ticket']:
    ticket_count = len([j for j, ticket in enumerate(dataset_v2['Ticket'])\
                        if ticket == ticket_num])
    ticket_count_list.append(ticket_count)
ticket_num_counts['Count'] = ticket_count_list

# Include Ticket_Count in dataset_v2 dataframe
ticket_count_long_list = []
for ticket_num in dataset_v2['Ticket']:
    ticket_count = int(ticket_num_counts[ticket_num_counts['Ticket']\
                                         == ticket_num]['Count'])
    ticket_count_long_list.append(ticket_count)
dataset_v2['Ticket_Count'] = ticket_count_long_list

# Histogram of Ticket_Count
dataset_v2['Ticket_Count'].hist()

# Boxplot of Ticket_Count by Survived
dataset_v2.boxplot(column='Ticket_Count', by='Survived')
plt.title('')
plt.ylabel('Ticket_Count')

# Variable: Fare
# Histogram of Fare
dataset_v2['Fare'].hist()

# Boxplot of Fare by Survived
dataset_v2.boxplot(column='Fare',by='Survived')
plt.title('')
plt.ylabel('Fare')

# Variable: Embarked
# Show percentages of Embarked
print('Percentages:')
print(dataset_v2['Embarked'].value_counts(normalize=True)*100)

# Crosstab plot of Embarked
embarked_crosstab = pd.crosstab(index=dataset_v2['Embarked'],
                                columns=dataset_v2['Survived'])
embarked_crosstab.plot(kind='bar',
                       stacked=False)

## Fit second model with some feature engineering

# Fill missing values for Age
# Compute Normal Distribution paramters for 'Miss'
salut_miss_age_list = [age for i, age in enumerate(dataset_v2['Age'])\
                       if dataset_v2['Salutation'][i] == 'Miss']
salut_miss_age_list = list(filter(lambda x: not np.isnan(x),
                                  salut_miss_age_list)) # Remove NaNs from list
salut_miss_age_mean = float(np.mean(np.array(salut_miss_age_list)))
salut_miss_age_sd = float(np.std(np.array(salut_miss_age_list)))
# Compute Normal Distribution paramters for 'Mr'
salut_mr_age_list = [age for i, age in enumerate(dataset_v2['Age'])\
                       if dataset_v2['Salutation'][i] == 'Mr']
salut_mr_age_list = list(filter(lambda x: not np.isnan(x),
                                  salut_mr_age_list)) # Remove NaNs from list
salut_mr_age_mean = float(np.mean(np.array(salut_mr_age_list)))
salut_mr_age_sd = float(np.std(np.array(salut_mr_age_list)))
# Compute Normal Distribution paramters for 'Mrs'
salut_mrs_age_list = [age for i, age in enumerate(dataset_v2['Age'])\
                       if dataset_v2['Salutation'][i] == 'Mrs']
salut_mrs_age_list = list(filter(lambda x: not np.isnan(x),
                                  salut_mrs_age_list)) # Remove NaNs from list
salut_mrs_age_mean = float(np.mean(np.array(salut_mrs_age_list)))
salut_mrs_age_sd = float(np.std(np.array(salut_mrs_age_list)))
# Compute Mean Age for all passengers
overall_age_mean = float(np.mean(dataset_v2['Age']))
# Fill in missing values for age
import random