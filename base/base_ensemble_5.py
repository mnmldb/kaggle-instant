#based on the following kernel: https://www.kaggle.com/dimitreoliveira/ensembling-and-evaluating-magic-models  

import numpy as np
import pandas as pd 
from sklearn.svm import SVC, NuSVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

#sampling for code check -- need to remove when formal submission
sample_group = [i for i in range(10)]
df_train = df_train[df_train['wheezy-copper-turtle-magic'].isin(sample_group)]
df_test = df_test[df_test['wheezy-copper-turtle-magic'].isin(sample_group)]
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

#set number of folds
N_FOLDS = 5

#extract feature names
features = [c for c in df_train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
num_groups = len(df_train['wheezy-copper-turtle-magic'].unique())

#create empty list for out of fold
num_models = 5
model_names = ['knn', 'mlp', 'svc', 'nusvc', 'lr']
oof = np.zeros((len(df_train), num_models))
predictions = np.zeros((len(df_test), num_models))

#set parameters
param_pca = {'n_components':'mle',
             'svd_solver':'full',
             'random_state':4}

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

#build 512 models
for i in range(num_groups):
    print('wheezy-copper-turtle-magic {}\n'.format(i))
    
    #extract subset of dataset where wheezy-copper-turtle-magic equals to i
    X_train = df_train[df_train['wheezy-copper-turtle-magic'] == i][features]
    y_train = df_train[df_train['wheezy-copper-turtle-magic'] == i]['target']
    X_test = df_test[df_test['wheezy-copper-turtle-magic'] == i][features]
    X_train_idx = X_train.index
    X_test_idx = X_test.index

    #reset subset index for k-fold
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    #concat (vertical) X_train and X_test for PCA and tandard scaling
    X_concat = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], axis=0)
    X_concat_scaled = StandardScaler().fit_transform(PCA(**param_pca).fit_transform(X_concat))
    X_train_scaled = X_concat_scaled[:X_train.shape[0]]
    X_test_scaled = X_concat_scaled[X_train.shape[0]:]

    #stratified k-fold
    skf = StratifiedKFold(n_splits=N_FOLDS, random_state=0)
    counter = 0

    for trn_idx, val_idx in skf.split(X_train_scaled, y_train):
        trn_x = X_train_scaled[trn_idx, :] #numpy
        val_x = X_train_scaled[val_idx, :] #numpy
        trn_y = y_train[trn_idx]

        counter += 1
        print('Fold {}\n'.format(counter))
        models = [KNeighborsClassifier(**param_knn),
                  MLPClassifier(**param_mlp),
                  SVC(**param_svc),
                  NuSVC(**param_nusvc),
                  LogisticRegression(**param_lr)]
        
        for i in range(num_models):
            model = models[i]
            model.fit(trn_x, trn_y)
            
            #oof: val_idx is inside of X_train_idx (1 of 512 groups)
            #predictions: predict X_test_idx (1 of 512 groups)
            oof[X_train_idx[val_idx], i] = model.predict_proba(val_x)[:, 1]
            predictions[X_test_idx, i] += model.predict_proba(X_test_scaled)[:, 1] / N_FOLDS
            
#print single model score
for i in range(num_models):
    score = roc_auc_score(df_train['target'], oof[:, i])
    print('{}: {}'.format(model_names[i], score))

#export single model file
sub = pd.read_csv('../input/sample_submission.csv')
sub = sub[:len(df_test)] #nothing happens if using all data
for i, model in enumerate(model_names):
    sub['target'] = predictions[:, i]
    sub.to_csv('submission_' + model + '.csv', index=False)