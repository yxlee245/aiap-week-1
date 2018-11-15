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

# Histograms for Age by Mr, Mrs, Miss, Master
salut_master_age_list = [age for i, age in enumerate(dataset_v2['Age'])\
                       if dataset_v2['Salutation'][i] == 'Master']
salut_master_age_list = list(filter(lambda x: not np.isnan(x),
                                  salut_master_age_list)) # Remove NaNs from list
salut_miss_age_list = [age for i, age in enumerate(dataset_v2['Age'])\
                       if dataset_v2['Salutation'][i] == 'Miss']
salut_miss_age_list = list(filter(lambda x: not np.isnan(x),
                                  salut_miss_age_list)) # Remove NaNs from list
salut_mr_age_list = [age for i, age in enumerate(dataset_v2['Age'])\
                       if dataset_v2['Salutation'][i] == 'Mr']
salut_mr_age_list = list(filter(lambda x: not np.isnan(x),
                                  salut_mr_age_list)) # Remove NaNs from list
salut_mrs_age_list = [age for i, age in enumerate(dataset_v2['Age'])\
                       if dataset_v2['Salutation'][i] == 'Mrs']
salut_mrs_age_list = list(filter(lambda x: not np.isnan(x),
                                  salut_mrs_age_list)) # Remove NaNs from list
plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.hist([salut_master_age_list])
plt.title('Salutation: Master')
plt.subplot(222)
plt.hist([salut_miss_age_list])
plt.title('Salutation: Miss')
plt.subplot(223)
plt.hist([salut_mr_age_list])
plt.title('Salutation: Mr')
plt.subplot(224)
plt.hist([salut_mrs_age_list])
plt.title('Salutation: Mrs')

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
# Compute Uniform Distribution paramters for 'Master'
salut_master_age_min = min(salut_master_age_list)
salut_master_age_max = max(salut_master_age_list)
# Compute Normal Distribution paramters for 'Miss'
salut_miss_age_mean = float(np.mean(salut_miss_age_list))
salut_miss_age_sd = float(np.std(np.array(salut_miss_age_list)))
# Compute Normal Distribution paramters for 'Mr'
salut_mr_age_mean = float(np.mean(np.array(salut_mr_age_list)))
salut_mr_age_sd = float(np.std(np.array(salut_mr_age_list)))
# Compute Normal Distribution paramters for 'Mrs'
salut_mrs_age_mean = float(np.mean(np.array(salut_mrs_age_list)))
salut_mrs_age_sd = float(np.std(np.array(salut_mrs_age_list)))
# Compute Mean Age for all passengers
overall_age_mean = float(np.mean(dataset_v2['Age']))

# Fill in missing values for age
import random
random.seed(2018)
for i, salutation in enumerate(dataset_v2['Salutation']):
    try:
        age = 0
        if np.isnan(dataset_v2['Age'][i]):
            if salutation == 'Master':
                age = \
                random.uniform(salut_master_age_min,salut_master_age_max)
            elif salutation == 'Miss':
                age = \
                random.normalvariate(salut_miss_age_mean, salut_miss_age_sd)
            elif salutation == 'Mr':
                age = \
                random.normalvariate(salut_mr_age_mean, salut_mr_age_sd)
            elif salutation == 'Mrs':
                age = \
                random.normalvariate(salut_mrs_age_mean, salut_mrs_age_sd)
            else:
                age = overall_age_mean
        # Ensure that age is non-negative
        if age < 0:
            age = 0
        dataset_v2['Age'][i] = age
    except TypeError: # No NaN in Age field
        continue

# Determine mode for Embarked
embarked_counts = np.unique(dataset_v2['Embarked'].dropna(inplace=False),
                            return_counts=True)
embarked_mode = embarked_counts[0][np.argmax(embarked_counts[1])]

# Fill in missing values for Embarked using mode
for i, embarked in enumerate(dataset_v2['Embarked']):
    try:
        if np.isnan(embarked):
            dataset_v2['Embarked'][i] = embarked_mode
    except TypeError:
        continue
    
# Encode categorical variables
dataset_v2 = pd.get_dummies(dataset_v2, columns=['Sex', 'Embarked'],
                            drop_first=True)


# Fit model version 2
# Split dataset to training and test sets
X = dataset_v2.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Salutation', 'Survived'],
                    axis=1)
y = dataset_v2['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=2018, stratify=y)
# Train model v2
rf_v2 = RandomForestClassifier(random_state=2018)
rf_v2.fit(X_train, y_train)
print('Accuracy:', accuracy_score(y_test, rf_v2.predict(X_test)))

# Feature Importance Plot
feature_importances = pd.DataFrame(rf_v2.feature_importances_,
                                   index=X_train.columns,
                                   columns=['importance']).\
                                   sort_values('importance', ascending=False)
plt.figure()
plt.barh(feature_importances.index, feature_importances['importance'])
plt.gca().invert_yaxis()

# Partial Dependence Plots
from pdpbox import pdp
# Plot PDP for Fare
pdp_fare = pdp.pdp_isolate(model=rf_v2, dataset=X_train.assign(Survived=y_train),
                        model_features=list(X_train.columns),feature='Fare',
                        num_grid_points=20)
pdp.pdp_plot(pdp_isolate_out=pdp_fare, feature_name='Fare');
# Plot PDP for Sex
pdp_sex = pdp.pdp_isolate(model=rf_v2, dataset=X_train.assign(Survived=y_train),
                        model_features=list(X_train.columns),feature='Sex_male',
                        num_grid_points=20)
pdp.pdp_plot(pdp_isolate_out=pdp_sex, feature_name='Sex');
# Plot PDP for Pclass
pdp_pclass = pdp.pdp_isolate(model=rf_v2, dataset=X_train.assign(Survived=y_train),
                        model_features=list(X_train.columns),feature='Pclass',
                        num_grid_points=20)
pdp.pdp_plot(pdp_isolate_out=pdp_pclass, feature_name='Pclass');
# Plot PDP for Ticket_Count
pdp_ticket_count = pdp.pdp_isolate(model=rf_v2, dataset=X_train.assign(Survived=y_train),
                        model_features=list(X_train.columns),feature='Ticket_Count',
                        num_grid_points=20)
pdp.pdp_plot(pdp_isolate_out=pdp_ticket_count, feature_name='Ticket_Count');
# Plot PDP for Num_Relative
pdp_num_relative = pdp.pdp_isolate(model=rf_v2, dataset=X_train.assign(Survived=y_train),
                        model_features=list(X_train.columns),feature='Num_Relative',
                        num_grid_points=20)
pdp.pdp_plot(pdp_isolate_out=pdp_num_relative, feature_name='Num_Relative');
# Plot PDP for Age
pdp_age = pdp.pdp_isolate(model=rf_v2, dataset=X_train.assign(Survived=y_train),
                        model_features=list(X_train.columns),feature='Age',
                        num_grid_points=20)
pdp.pdp_plot(pdp_isolate_out=pdp_age, feature_name='Age');
# Plot PDP for Embarked
pdp_embarked = pdp.pdp_isolate(model=rf_v2, dataset=X_train.assign(Survived=y_train),
                        model_features=list(X_train.columns),
                        feature=['Embarked_S', 'Embarked_Q'],
                        num_grid_points=20)
pdp.pdp_plot(pdp_isolate_out=pdp_embarked, feature_name='Embarked');

## Fit third model after more feature engineering
dataset_v3 = dataset_v2.copy()

# Feature Engineering
# Compute Fare_Class_Indicator
dataset_v3['Fare_Class_Indicator'] = dataset_v3['Fare'] / dataset_v3['Pclass']
# Convert Age to categorical variable (Age_Cat)
# 0 - Age <= 35, 1 - Age > 35
dataset_v3['Age_Cat'] = (dataset_v3['Age'] > 35).astype(int)
# Convert Num_Relative to categorical variable (Family_Size_Cat)
# 0 - Num_Relative <= 4, 1 - Num_Relative > 4
dataset_v3['Family_Size_Cat'] = (dataset_v3['Num_Relative'] > 4).astype(int)

# Fit model version 3
# Split dataset to training and test sets
X = dataset_v3.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin',
                     'Salutation', 'Survived', 'Fare', 'Pclass',
                     'Ticket_Count', 'Age', 'Num_Relative',
                     'Embarked_Q', 'Embarked_S'],
                    axis=1)
y = dataset_v3['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=2018, stratify=y)
# Train model v3
rf_v3 = RandomForestClassifier(random_state=2018)
rf_v3.fit(X_train, y_train)
print('Accuracy: {:.1f}%'.format(accuracy_score(y_test, rf_v3.predict(X_test)) * 100))

# Feature Importance Plot
feature_importances = pd.DataFrame(rf_v3.feature_importances_,
                                   index=X_train.columns,
                                   columns=['importance']).\
                                   sort_values('importance', ascending=False)
plt.figure()
plt.barh(feature_importances.index, feature_importances['importance'])
plt.gca().invert_yaxis()

# Partial Dependence Plots
from pdpbox import pdp
# Plot PDP for Fare_Class_Indicator
pdp_fci = pdp.pdp_isolate(model=rf_v3, dataset=X_train.assign(Survived=y_train),
                        model_features=list(X_train.columns),
                        feature='Fare_Class_Indicator', num_grid_points=20)
pdp.pdp_plot(pdp_isolate_out=pdp_fci, feature_name='Fare_Class_Indicator');
# Plot PDP for Sex
pdp_sex = pdp.pdp_isolate(model=rf_v3, dataset=X_train.assign(Survived=y_train),
                        model_features=list(X_train.columns),
                        feature='Sex_male', num_grid_points=20)
pdp.pdp_plot(pdp_isolate_out=pdp_sex, feature_name='Sex');
# Plot PDP for Family_Size_Cat
pdp_sex = pdp.pdp_isolate(model=rf_v3, dataset=X_train.assign(Survived=y_train),
                        model_features=list(X_train.columns),
                        feature='Family_Size_Cat', num_grid_points=20)
pdp.pdp_plot(pdp_isolate_out=pdp_sex, feature_name='Family_Size_Cat');
# Plot PDP for Age_Cat
pdp_sex = pdp.pdp_isolate(model=rf_v3, dataset=X_train.assign(Survived=y_train),
                        model_features=list(X_train.columns),
                        feature='Age_Cat', num_grid_points=20)
pdp.pdp_plot(pdp_isolate_out=pdp_sex, feature_name='Age_Cat');

### Model Re-Training and Fine-Tuning ###
# Fit model version 3
# Split dataset to training and test sets
X = dataset_v3[['Fare_Class_Indicator', 'Sex_male', 'Family_Size_Cat',
                'Age_Cat']]
y = dataset_v3['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=2018, stratify=y)
# Train model v3
rf_v3 = RandomForestClassifier(random_state=2018)
rf_v3.fit(X_train, y_train)
print('Accuracy: {:.1f}%'.format(accuracy_score(y_test, rf_v3.predict(X_test)) * 100))

# Model tuning
# Plot of oob_score vs max_depth
oob_score_list = []
for max_depth in range(1, 11):
    rf_tuning = RandomForestClassifier(max_depth=max_depth, oob_score=True,
                                       random_state=2018)
    rf_tuning.fit(X_train, y_train)
    oob_score_list.append(rf_tuning.oob_score_)
print('max_depth with lowest oob_score:', oob_score_list.index(min(oob_score_list)) + 1)
plt.plot([i for i in range(1, 11)], oob_score_list);
# Plot of oob_score vs max_leaf_nodes
oob_score_list = []
for max_leaf_nodes in range(2, 16):
    rf_tuning = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,
                                       oob_score=True,
                                       random_state=2018)
    rf_tuning.fit(X_train, y_train)
    oob_score_list.append(rf_tuning.oob_score_)
print('max_leaf_nodes with lowest oob_score:', oob_score_list.index(min(oob_score_list)) + 2)
plt.plot([i for i in range(2, 16)], oob_score_list);
# Plot of oob_score vs max_features
oob_score_list = []
max_features_list = ['sqrt', 'log2']
for max_features in max_features_list:
    rf_tuning = RandomForestClassifier(max_features=max_features,
                                       oob_score=True,
                                       random_state=2018)
    rf_tuning.fit(X_train, y_train)
    oob_score_list.append(rf_tuning.oob_score_)
#print('max_features with lowest oob_score:',
#      max_features_list[oob_score_list.index(min(oob_score_list))])
plt.plot([i for i in range(len(max_features_list))], oob_score_list)
plt.xticks([0, 1], max_features_list);
# Plot of oob_score vs n_estimators
oob_score_list = []
n_estimators_list = [100, 300, 500, 700, 900]
for n_estimators in n_estimators_list:
    rf_tuning = RandomForestClassifier(n_estimators=n_estimators,
                                       oob_score=True,
                                       random_state=2018)
    rf_tuning.fit(X_train, y_train)
    oob_score_list.append(rf_tuning.oob_score_)
print('n_estimators with lowest oob_score:',
      n_estimators_list[oob_score_list.index(min(oob_score_list))])
plt.plot(n_estimators_list, oob_score_list);

# Fit tuned model
rf_tuned = RandomForestClassifier(max_depth=5,
                                  max_leaf_nodes=70,
                                  max_features='sqrt',
                                  n_estimators=300,
                                  random_state=2018)
rf_tuned.fit(X_train, y_train)
print('Accuracy: {:.1f}%'.format(accuracy_score(y_test, rf_tuned.predict(X_test)) * 100))

# Subsampling
from sklearn.ensemble import forest

# Define functions for subsampling
def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))

# Enable subsampling for various sizes of subsampling
accuracy_score_list = []
fraction_list = [0.3, 0.4, 0.5, 0.6, 0.7]
for fraction in fraction_list:
    set_rf_samples(int(fraction * len(X_train)))

    # Fit tuned model
    rf_tuned = RandomForestClassifier(max_depth=5,
                                      max_leaf_nodes=70,
                                      max_features='sqrt',
                                      n_estimators=300,
                                      random_state=2018)
    rf_tuned.fit(X_train, y_train)
    accuracy_score_list.append(accuracy_score(y_test, rf_tuned.predict(X_test)))

print('Fraction with highest accuracy:',
      fraction_list[accuracy_score_list.index(max(accuracy_score_list))])

# Train model with 40% subsampling
set_rf_samples(int(0.4 * len(X_train)))
rf_tuned = RandomForestClassifier(max_depth=5,
                                  max_leaf_nodes=70,
                                  max_features='sqrt',
                                  n_estimators=300,
                                  random_state=2018)
rf_tuned.fit(X_train, y_train)
print('Accuracy: {:.1f}%'.format(accuracy_score(y_test, rf_tuned.predict(X_test)) * 100))

# Train model using full dataset
rf_final = RandomForestClassifier(max_depth=5,
                                  max_leaf_nodes=70,
                                  max_features='sqrt',
                                  n_estimators=300,
                                  random_state=2018)
rf_final.fit(X, y)

# Disable subsampling
reset_rf_samples()

### Prediction on Test set ###
# Import test set
kaggle_testset_raw = pd.read_csv('data/titanic_test.csv')
kaggle_testset_v1 = kaggle_testset_raw.copy()

# Extract Salutation from Name
kaggle_testset_v1['Salutation'] = salutation_list_create(kaggle_testset_v1['Name'])
kaggle_testset_v1['Salutation'].value_counts()

# Fill in missing values for Age
import random
random.seed(2018)
for i, salutation in enumerate(kaggle_testset_v1['Salutation']):
    try:
        age = 0
        if np.isnan(kaggle_testset_v1['Age'][i]):
            if salutation == 'Master':
                age = \
                random.uniform(salut_master_age_min,salut_master_age_max)
            elif salutation == 'Miss':
                age = \
                random.normalvariate(salut_miss_age_mean, salut_miss_age_sd)
            elif salutation == 'Mr':
                age = \
                random.normalvariate(salut_mr_age_mean, salut_mr_age_sd)
            elif salutation == 'Mrs':
                age = \
                random.normalvariate(salut_mrs_age_mean, salut_mrs_age_sd)
            else:
                age = overall_age_mean
        # Ensure that age is non-negative
        if age < 0:
            age = 0
        kaggle_testset_v1['Age'][i] = age
    except TypeError: # No NaN in Age field
        continue

# Fill in missing value for Fare
overall_fare_mean = float(np.mean(dataset_v3['Fare']))
for i, fare in enumerate(kaggle_testset_v1['Fare']):
    try:
        if np.isnan(kaggle_testset_v1['Fare'][i]):
            kaggle_testset_v1['Fare'][i] = overall_fare_mean
    except:
        continue
    
# Compute Fare_Class_Indicator
kaggle_testset_v1['Fare_Class_Indicator'] = kaggle_testset_v1['Fare'] / kaggle_testset_v1['Pclass']

# Create Family_Size_Cat
kaggle_testset_v1['Family_Size'] = kaggle_testset_v1['SibSp'] + kaggle_testset_v1['Parch']
kaggle_testset_v1['Family_Size_Cat'] = (kaggle_testset_v1['Family_Size'] > 4).astype(int)

# Create Age_Cat
kaggle_testset_v1['Age_Cat'] = (kaggle_testset_v1['Age'] > 35).astype(int)

# Encode Sex
kaggle_testset_v1 = pd.get_dummies(kaggle_testset_v1, columns=['Sex'],
                                   drop_first=True)

# Obtain predictions
X_kaggle = kaggle_testset_v1[['Fare_Class_Indicator', 'Sex_male',
                              'Family_Size_Cat', 'Age_Cat']]
y_pred = rf_final.predict(X_kaggle)

# Create DataFrame with PassengerId and Predictions
predictions = pd.DataFrame({
        'PassengerId': kaggle_testset_v1['PassengerId'],
        'Survived': y_pred
        })