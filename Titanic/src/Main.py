# For more explanations to this script take a look at the Jupyter Notebook one folder above

import sys
sys.path.insert(0, '../../../Utilities') # https://github.com/LarsNeR/Utilities
from Utilities import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv('../data/train.csv')
df_train.head()

print(get_highly_correlating_columns(df_train, threshold=0.5))
print(get_sparsely_filled_columns(df_train, threshold=0.2))
print("Any na in Pclass: " + str(df_train['Pclass'].isna().any()))
print("Any na in Name: " + str(df_train['Name'].isna().any()))
print("Any na in Sex: " + str(df_train['Sex'].isna().any()))
print("Any na in Age: " + str(df_train['Age'].isna().any()))
print("Any na in SibSp: " + str(df_train['SibSp'].isna().any()))
print("Any na in Parch: " + str(df_train['Parch'].isna().any()))
print("Any na in Ticket: " + str(df_train['Ticket'].isna().any()))
print("Any na in Fare: " + str(df_train['Fare'].isna().any()))
print("Any na in Cabin: " + str(df_train['Cabin'].isna().any()))
print("Any na in Embarked: " + str(df_train['Embarked'].isna().any()))


### PassengerId
print(df_train['PassengerId'].values[:200])

### Pclass
df_train.groupby('Pclass')['Survived'].agg(np.mean).plot(kind='bar', title='Survival rate by Pclass')
plt.show()
get_categorical_distrbution(df_train['Pclass'])

### Name
df_train.loc[:20, 'Name']

### Sex
df_train.groupby('Sex')['Survived'].agg(np.mean).plot(kind='bar', title='Survival rate by sex')
plt.show()
get_categorical_distrbution(df_train['Sex'])

df_train['Sex'].unique()

### Age
bins = pd.cut(df_train['Age'], 6)
df_train.groupby(bins)['Survived'].agg(np.mean).plot(kind='bar', title='Survival rate by age')
plt.show()
get_numerical_distrbution(df_train['Age'])

### SibSp
df_train.groupby('SibSp')['Survived'].agg(np.mean).plot(kind='bar', title='Survival rate by number of SibSp')
plt.show()
get_categorical_distrbution(df_train['SibSp'])

### Parch
avg_by_Pclass = df_train.groupby('Parch')['Survived'].agg(np.mean).plot(kind='bar', title='Survival rate by number of Parch')
plt.show()
get_categorical_distrbution(df_train['Parch'])

### Ticket
df_train.loc[:20, 'Ticket']

### Fare
bins = pd.cut(df_train['Fare'], 6)
df_train.groupby(bins)['Survived'].agg(np.mean).plot(kind='bar', title='Survival rate by fare')
plt.show()
get_numerical_distrbution(df_train['Fare'])

### Cabin
df_train.loc[:20, 'Cabin']

### Embarked
df_train.groupby('Embarked')['Survived'].agg(np.mean).plot(kind='bar', title='Survival rate by Embarked')
plt.show()
get_categorical_distrbution(df_train['Embarked'])

### Insights
# - No highly correlating features
# - Cabin has many NaNs (Age and Embarked have some)
# - PassengerId is just an incrementing number. Only important for test set
# - Passengers having a higher socio-economic status have a higher survival rate
# - We could extract information about titles and family relations from names
# - Sex feature is binary
# - Having more male passengers, female passengers have survived more often than the male ones
# - Survival rate depends on age. A good binning has to be found
# - Estimation of age (xx.5) can be important, estimating with help of SibSp could be possible
# - Number of SibSp has a small impact on the survival rate
# - Number of Parch has a small impact on the survival rate. Imbalanced feature (over 75% have 0 Parch)
# - Ticket seems to be an arbitrary Code. Maybe it correlates in any kind with the survival rate
# - Survival rate depends on fare. A good binning has to be found
# - Check if the cabin letter/number correlates in any way with the survival rate
# - Survival rate slightly depends on Embarked

## Feature Generation
### PassengerId
def drop_passenger_id(df):
    return df.drop('PassengerId', axis=1, errors='ignore')

### Name
titles = "Master|Don|Rev|Dr"
def check_if_title(df):
    return df['Name'].str.contains(titles, regex=True)

### Sex
def get_converted_sex(df):
    return df['Sex'].astype('category').cat.codes


### Age
def fillna_age(df):
    return df['Age'].fillna(df['Age'].mean())

def bin_age(df):
    return pd.cut(df['Age'], [0, 14, 25, 50, 100], labels=[1,2,3,4])

### Fare
def fillna_fare(df):
    return df['Fare'].fillna(df['Fare'].mean())

def bin_fare(df):
    return pd.cut(df['Fare'], [-1, 30, 100, 150, 1000], labels=[1,2,3,4])

### Cabin
def drop_cabin(df):
    return df.drop('Cabin', axis=1, errors='ignore')

### Embarked
def one_hot_encode_embarked(df):
    df['EmbarkS'] = (df['Embarked'] == 'S').astype('category').cat.codes
    df['EmbarkC'] = (df['Embarked'] == 'C').astype('category').cat.codes
    df['EmbarkQ'] = (df['Embarked'] == 'Q').astype('category').cat.codes
    return df

### Add FamilySize
def get_family_size(df):
    return df['SibSp'] + df['Parch']

### Check if is alone(df):
def is_alone(df):
    df_size = df['SibSp'] + df['Parch']
    df_size = df_size == 0
    return df_size

### Preprocessing
def preprocess_df(df):
    df = drop_passenger_id(df)
    df['HasTitle'] = check_if_title(df)
    df['SexBinary'] = get_converted_sex(df)
    df['Age'] = fillna_age(df)
    df['AgeBin'] = bin_age(df)
    df['Fare'] = fillna_fare(df)
    df['FareBin'] = bin_fare(df)
    df = drop_cabin(df)
    df = one_hot_encode_embarked(df)
    df['FamilySize'] = get_family_size(df)
    df['IsAlone'] = is_alone(df)
    df = df.drop(df[['Name', 'Sex', 'Age', 'Ticket', 'Fare', 'Embarked']].columns, axis=1, errors='ignore')
    return df

## Training
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
df = preprocess_df(df_train)
y = df['Survived'].values
X = df.drop('Survived', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y)

### MLP
from sklearn.neural_network import MLPClassifier
clf_mlp = MLPClassifier()
clf_mlp.fit(X_train, y_train)
y_pred = clf_mlp.predict(X_test)
print("F1-Score: " + str(f1_score(y_test, y_pred)))

### Logistic Regression
from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()
clf_lr.fit(X_train, y_train)
y_pred = clf_lr.predict(X_test)
print("F1-Score: " + str(f1_score(y_test, y_pred)))

### Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def train_model(X_train, y_train, params, verbose=0):
    clf = Sequential()

    clf.add(Dense(units = X_train.shape[1]*2, activation='relu', input_dim = X_train.shape[1]))
    clf.add(Dropout(float(params['dropout'])))
    
    clf.add(Dense(units = int(X_train.shape[1]*1.5), activation='relu'))
    clf.add(Dropout(float(params['dropout'])))
    
    clf.add(Dense(units = 1, activation='sigmoid'))

    clf.compile(optimizer = params['optimizer'], loss = params['losses'], metrics=['accuracy'])

    history = clf.fit(X_train, y_train, epochs = int(params['epochs']), batch_size = int(params['batch_size']), verbose=verbose)
    return history, clf

p = {'batch_size': 10,
     'epochs': 150,
     'dropout': 0.01,
     'optimizer': 'rmsprop',
     'losses': 'binary_crossentropy'
    }
h, clf_nn = train_model(X_train, y_train, p, 0)
y_pred = clf_nn.predict(X_test)
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0
print("F1-Score: " + str(f1_score(y_test, y_pred)))

### Convert to Kaggle-Submission
df_test = pd.read_csv('../data/test.csv')
df_result = pd.DataFrame({'PassengerId': df_test['PassengerId']})
df = preprocess_df(df_test)
X_test = df.values

clf_mlp.fit(X, y)
df_result['Survived'] = clf_mlp.predict(X_test)
df_result.to_csv('../data/result_mlp.csv', index=False)

clf_lr.fit(X, y)
df_result['Survived'] = clf_lr.predict(X_test)
df_result.to_csv('../data/result_lr.csv', index=False)

clf_nn.fit(X, y)
y_pred_nn = clf_nn.predict(X_test)
y_pred_nn[y_pred_nn > 0.5] = 1
y_pred_nn[y_pred_nn <= 0.5] = 0
df_result['Survived'] = y_pred_nn
df_result['Survived'] = df_result['Survived'].astype(int)
df_result.to_csv('../data/result_nn.csv', index=False)


#### Submitting them to Kaggle gave following results
# - MLP: 0.803 (Rank 1319)
# - Logistic Regression: 0.7751 (Rank 6055)
# - NN: 0.7655 (Rank ?)

### Hyperparameter-Tuning
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
df = preprocess_df(df_train)
parameters = {
    'hidden_layer_sizes':[(24, 48, 12), (48, 96, 96, 48)], 
    'activation':['tanh'], 
    'alpha': [0.001, 0.005], 
    'learning_rate': ['adaptive']}
y = df['Survived'].values
X = df.drop('Survived', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y)
mlp = MLPClassifier()
clf = GridSearchCV(mlp, parameters, cv=int(X_train.shape[0]/10), verbose=2, n_jobs=-1)
clf.fit(X_train, y_train)
print(clf.best_params_)
y_pred = clf.predict(X_test)
print("F1-Score: " + str(f1_score(y_test, y_pred)))

## Final run of MLP with these parameters
from sklearn.neural_network import MLPClassifier
df_test = pd.read_csv('../data/test.csv')
df_result = pd.DataFrame({'PassengerId': df_test['PassengerId']})
df = preprocess_df(df_test)
X_test = df.values
clf_mlp = MLPClassifier(hidden_layer_sizes=(24, 48, 12), solver='adam', max_iter=200, learning_rate='adaptive', alpha=0.005, activation='tanh')
clf_mlp.fit(X, y)
df_result['Survived'] = clf_mlp.predict(X_test)
df_result.to_csv('../data/result_final.csv', index=False)


# Gives a result of 0.79904
# 
# Seems like the 0.803 was a lucky shot