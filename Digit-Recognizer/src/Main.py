# For more explanations to this script take a look at the Jupyter Notebook one folder above

import sys
sys.path.insert(0, '../Utilities') # https://github.com/LarsNeR/Utilities
from Utilities import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

df_train = pd.read_csv('Digit-Recognizer/data/train.csv')
df_test = pd.read_csv('Digit-Recognizer/data/test.csv')
df_train.head()

first_image = df_train.iloc[0, 1:].values
first_image = first_image
first_image = np.reshape(first_image, (28,28)).astype(np.uint8)
img = Image.fromarray(first_image, 'L')
plt.imshow(img, cmap='gray')


### Feature Exploration
print(get_sparsely_filled_columns(df_train, threshold=0))
print(get_sparsely_filled_columns(df_test, threshold=0))
# No NaNs

## Preprocessing
def preprocess_df(df):
    df = df/255
    return df

## Training
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
df_X = df_train.iloc[:, 1:]
df = preprocess_df(df_X)
X_train = df.values
y_train = df_train.iloc[:, 0].values
kFold = StratifiedKFold(n_splits=10, shuffle=True)

p = {'batch_size': 100,
     'epochs': 3,
     'dropout': 0.3,
     'optimizer': 'rmsprop',
     'losses': 'categorical_crossentropy',
     '1st_kernel_size': (2,2),
     '2nd_kernel_size': (3,3)
    }

### Keras CNN 2D
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.layers import Dropout
from keras.utils import to_categorical

def train_model_cnn2d(X_train, y_train, X_val, y_val, params, verbose=0):
    clf = Sequential()

    clf.add(Conv2D(filters = 32, kernel_size = params['1st_kernel_size'], activation ='relu', input_shape = (28,28,1)))
    clf.add(Conv2D(filters = 32, kernel_size = params['1st_kernel_size'], activation ='relu'))
    clf.add(MaxPool2D(pool_size=(2,2)))
    clf.add(Dropout(float(params['dropout'])))

    clf.add(Conv2D(filters = 64, kernel_size = params['2nd_kernel_size'], activation ='relu'))
    clf.add(Conv2D(filters = 64, kernel_size = params['2nd_kernel_size'], activation ='relu'))
    clf.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    clf.add(Dropout(float(params['dropout'])))

    clf.add(Flatten())
    clf.add(Dense(256, activation = "relu"))
    clf.add(Dropout(float(params['dropout'])))
    clf.add(Dense(10, activation = "softmax"))
    
    clf.compile(optimizer=params['optimizer'], loss=params['losses'], metrics=["accuracy"])
    history = clf.fit(X_train, y_train, epochs=int(params['epochs']), batch_size=int(params['batch_size']), verbose=verbose)
    return history, clf

def cross_validation_cnn2d(X_train, y_train, params):
    accs = []
    X_train = np.reshape(X_train, (-1, 28, 28, 1))
    for train, test in kFold.split(X_train, y_train):
        y_ohe = to_categorical(y_train, num_classes=10)
        _, clf = train_model_cnn2d(X_train[train,:], y_ohe[train,:], None, None, params, 1)
        y_pred = clf.predict(X_train[test])
        accs.append(accuracy_score(np.argmax(y_pred, axis=1), y_train[test]))
    print(np.mean(accs))
    return train_model_cnn2d(X_train, to_categorical(y_train, num_classes=10), None, None, params, 1)

_, clf_cnn2d = cross_validation_cnn2d(X_train, y_train, p)

# Accuracy of around 98%

## Predicting
df_test = pd.read_csv('Digit-Recognizer/data/test.csv')
df_result = pd.DataFrame({'ImageId': np.arange(1, df_test.shape[0]+1)})
df = preprocess_df(df_test)
X_test = df.values
X_test = np.reshape(X_test, (-1, 28, 28, 1))
df_result['Label'] = np.argmax(clf_cnn2d.predict(X_test), axis=1)
df_result.to_csv('Digit-Recognizer/data/result_cnn2d.csv', index=False)


#### Submitting them to Kaggle gave following results
# - NN: 0.9660 (Rank 2161)
# - CNN: 0.5862 (Rank ?)
# - CNN-2D: 0.9874 (Rank 1326)

### Hyperparameter-Tuning
import talos as ta
p = {'batch_size': [50],
     'epochs': [4],
     'dropout': [0.1, 0.3, 0.5, 0.7],
     'optimizer': ['rmsprop', 'nadam'],
     'losses': ['categorical_crossentropy'],
     '1st_kernel_size': [(2,2), (3,3)],
     '2nd_kernel_size': [(3,3), (4,4)]
    }

X_train = np.reshape(X_train, (-1, 28, 28, 1))
y_ohe = to_categorical(y_train, num_classes=10)
h = ta.Scan(X_train, y_ohe,
          params=p,
          dataset_name='Digit Recognizer',
          experiment_no='1',
          model=train_model_cnn2d,
          grid_downsample=0.5)
ta.Reporting(t).data


# Best parameters:
# 
# {'dropout': 0.1, 'optimizer': 'rmsprop', '1st_kernel_size': (3,3), '2nd_kernel_size': (4,4)}
# 
# Accuracy: 0.9907139
# 

## Final run of CNN-2D with these parameters
p = {'batch_size': 50,
     'epochs': 6,
     'dropout': 0.1,
     'optimizer': 'rmsprop',
     'losses': 'categorical_crossentropy',
     '1st_kernel_size': (3,3),
     '2nd_kernel_size': (4,4)
    }
X_train = np.reshape(X_train, (-1, 28, 28, 1))
y_ohe = to_categorical(y_train, num_classes=10)
_, clf_final = train_model_cnn2d(X_train, y_ohe,_, _, p,verbose=1)

df_test = pd.read_csv('Digit-Recognizer/data/test.csv')
df_result = pd.DataFrame({'ImageId': np.arange(1, df_test.shape[0]+1)})
df = preprocess_df(df_test)
X_test = df.values
X_test = np.reshape(X_test, (-1, 28, 28, 1))
df_result['Label'] = np.argmax(clf_final.predict(X_test), axis=1)
df_result.to_csv('Digit-Recognizer/data/result_final.csv', index=False)

# Gives a result of 0.99157 (Rank 997)