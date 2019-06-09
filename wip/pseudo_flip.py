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

# referring to the kernel:
# https://www.kaggle.com/yizhitao/flip-y-lb-0-9697

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from tqdm import tqdm_notebook
import warnings
import multiprocessing
from scipy.optimize import minimize  
warnings.filterwarnings('ignore')

# import dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#sampling for code check -- need to remove when formal submission
sample_group = [i for i in range(10)]
train = train[train['wheezy-copper-turtle-magic'].isin(sample_group)]
test = test[test['wheezy-copper-turtle-magic'].isin(sample_group)]
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# columns
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

# single QDA
oof = np.zeros(len(train))
preds = np.zeros(len(test))

for i in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)

    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
    data2 = VarianceThreshold(threshold=2).fit_transform(data[cols])

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

    skf = StratifiedKFold(n_splits=11, random_state=42)
    for train_index, test_index in skf.split(train2, train2['target']):

        clf = QuadraticDiscriminantAnalysis(0.5)
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

auc = roc_auc_score(train['target'], oof)
print(f'AUC: {auc:.5}')

# pseudo labeling
test['target'] = preds
test.loc[test['target'] > 0.99, 'target'] = 1
test.loc[test['target'] < 0.01, 'target'] = 0

# target flip
usefull_test = test[(test['target'] == 1) | (test['target'] == 0)]
new_train = pd.concat([train, usefull_test]).reset_index(drop=True)
new_train.loc[oof > 0.99, 'target'] = 1
new_train.loc[oof < 0.01, 'target'] = 0

# preparation for training
model_names = ['qda', 'knn', 'mlp', 'svc', 'nusvc', 'lr']
num_models = len(model_names)
index_used = [0, 1, 2, 3, 4, 5] #set index of model to be used

oof2 = np.zeros((len(train), num_models))
pred2 = np.zeros((len(test), num_models))

# set parameters
param_qda = {'reg_param': 0.5}
param_knn = {'n_neighbors':17,
             'p':2.9}
param_mlp = {'random_state':3, 
            'activation':'relu', 
            'solver':'lbfgs', 
            'tol':1e-06, 
            'hidden_layer_sizes':(250, )}
param_svc = {'probability':True,
             'kernel':'poly',
             'degree':4, 
             'gamma':'auto',
             'random_state':42}
param_nusvc = {'probability':True,
               'kernel':'poly',
               'degree':4,
               'gamma':'auto',
               'random_state':4,
               'nu':0.59,
               'coef0':0.053}
param_lr = {'solver':'saga',
            'penalty':'l1', 
            'C':0.1}

# train models
for i in tqdm_notebook(range(512)):

    train2 = new_train[new_train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train[train['wheezy-copper-turtle-magic']==i].index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)

    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
    data2 = StandardScaler().fit_transform(VarianceThreshold(threshold=2).fit_transform(data[cols]))
    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

    skf = StratifiedKFold(n_splits=11, random_state=42)
    for train_index, test_index in skf.split(train2, train2['target']):
        oof_test_index = [t for t in test_index if t < len(idx1)]
        
        clf = QuadraticDiscriminantAnalysis(0.5)
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        if len(oof_test_index) > 0:
            oof2[idx1[oof_test_index]] = clf.predict_proba(train3[oof_test_index,:])[:,1]
        preds2[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
        models = [QuadraticDiscriminantAnalysis(**param_qda), KNeighborsClassifier(**param_knn), MLPClassifier(**param_mlp), SVC(**param_svc), NuSVC(**param_nusvc), LogisticRegression(**param_lr)]
        
        for i in index_used:
            model = models[i]
            model.fit(train3[train_index,:],train2.loc[train_index]['target'])
            if len(oof_test_index) > 0:
                oof2[idx1[oof_test_index], i] = clf.predict_proba(train3[oof_test_index,:])[:,1]
            preds2[idx2, i] += clf.predict_proba(test3)[:,1] / skf.n_splits          

#print single model score
for i in index_used:
    score = roc_auc_score(train['target'], oof2[:, i])
    print('{}: {}'.format(model_names[i], score))

#export single model file
sub = pd.read_csv('../input/sample_submission.csv')
sub = sub[:len(test)] #nothing happens if using all data
for i in index_used:
    sub['target'] = preds2[:, i]
    sub.to_csv('submission_' + model_names[i] + '.csv', index=False)

#export ensemble model file
sub2 = pd.read_csv('../input/sample_submission.csv')
sub3 = pd.read_csv('../input/sample_submission.csv')
sub2['target'] = 0.5 + preds[:, 0] + 0.25 * preds[:, 3] + 0.25 *preds[:, 4]
sub3['target'] = 2 / 3 * preds[:, 0] + 1 / 3 * preds[:, 4]
sub2.to_csv('submission_2.csv', index=False)
sub3.to_csv('submission_3.csv', index=False)


