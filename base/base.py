# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import NuSVC
from tqdm import tqdm

#import raw data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

#create empty list for out of fold
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))

#extract feature names
features = [c for c in df_train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

#arrange data for training
X_train = df_train[features]
y_train = df_train['target']
X_test = df_test[features]

#set parameter for NuSVC
params = {'probability': True,
          'kernel': 'poly', 
          'degree': 4,
          'gamma': 'auto',
          'random_state': 4, 
          'nu': 0.59, 
          'coef0': 0.053}

#cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for train_idx, valid_idx in tqdm(cv.split(X_train, y_train)):
    trn_x = X_train.iloc[train_idx, :]
    val_x = X_train.iloc[valid_idx, :]
    trn_y = y_train[train_idx]
    val_y = y_train[valid_idx]
    clf = NuSVC(**params)
    clf.fit(trn_x, trn_y)
    oof[valid_idx] = clf.predict_proba(val_x)
    predictions += clf.predict_proba(X_test) / cv.n_splits

#check ROC
print('CV score: {}'.format(roc_auc_score(y_train, oof)))

#Submission
submisson = pd.read_csv('../input/sample_submission.csv')
submission['target'] = predictions