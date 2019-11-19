import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from pymfe.mfe import MFE
from tqdm import tqdm

from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import cohen_kappa_score, mean_squared_error, classification_report, accuracy_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced

import lightgbm as lgb

parser = argparse.ArgumentParser(description='Process params for metastream.')
parser.add_argument('--omega', type=int, help='train window size.')
parser.add_argument('--gamma', type=int, help='selection window size.')
parser.add_argument('--initial', type=int, help='initial data.')
parser.add_argument('--target', help='target label.')
parser.add_argument('--eval_metric', help='eval metric for metastream.')
parser.add_argument('--metay', help='metay label.', default='clf')

args = parser.parse_args()

metadf = []

metrics = {
    'acc': accuracy_score
}

lgb_params = {
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'objective': 'binary',
        'metric': 'auc',
        # 'metric': 'multi_error',
        # 'objective': 'multiclass',
        # 'num_class': 3,
        'is_unbalance': True,
        'seed': 42,
        'verbosity': -1,
    }

params = [
        {"C": [1,10,100],
         "kernel": ["rbf", "linear", "poly", "sigmoid"]},
        {"max_depth": [3, None],
         "n_estimators": [100, 200, 300, 500],
         "max_features": stats.randint(1, 9),
         "min_samples_split": stats.randint(2, 11),
         "bootstrap": [True, False],
         "criterion": ["gini", "entropy"]}]

models = [
        SVC(),
        RandomForestClassifier(random_state=42)]

omega = args.omega
gamma = args.gamma
initial_data = args.initial
target = args.target
eval_metric = metrics[args.eval_metric]
metay_label = args.metay

df = pd.read_csv('../data/elec2/eletricity.csv')

datasize = omega * initial_data // gamma - 1 # initial base data
Xcv, ycv = df.loc[:datasize].drop([target], axis=1), df.loc[:datasize, target]

for model, param in tqdm(zip(models, params), total=len(models)):
    rscv = RandomizedSearchCV(model, param, n_jobs=-1,
                              scoring=make_scorer(eval_metric))
    rscv.fit(Xcv, ycv)
    model.set_params(**rscv.best_params_)

mfe = MFE()
for idx in tqdm(range(initial_data)):
    train = df.iloc[idx * gamma:idx * gamma + omega]
    sel = df.iloc[idx * gamma + omega:(idx+1) * gamma + omega]

    xtrain, ytrain = train.drop(target, axis=1), train[target]
    xsel, ysel = sel.drop(target, axis=1), sel[target]

    mfe.fit(xtrain.values, ytrain.values)
    ft = mfe.extract()
    mfe_feats = dict(zip(*ft))

    for model in models:
        model.fit(xtrain, ytrain)
    preds = [model.predict(xsel) for model in models]
    scores = [eval_metric(ysel, pred) for pred in preds]
    max_score = np.argmax(scores)
    mfe_feats[metay_label] = max_score

    metadf.append(mfe_feats)


metadf = pd.DataFrame(metadf)
metadf.to_csv('../data/elec2/meta.csv', index=False)


missing_columns = metadf.columns[metadf.isnull().any()].values
metadf.drop(columns=missing_columns, inplace=True)

mxtrain, mxtest, mytrain, mytest =\
    train_test_split(metadf.drop(metay_label, axis=1),
                     metadf[metay_label], random_state=42)


metas = lgb.train(lgb_params, train_set=lgb.Dataset(mxtrain, mytrain))

myhattest = np.argmax(metas.predict(mxtest), axis=1)
print("Kappa: ", cohen_kappa_score(mytest, myhattest))
print("GMean: ", geometric_mean_score(mytest, myhattest))
print("Accuracy: ", accuracy_score(mytest, myhattest))
print(classification_report(mytest, myhattest))
print(classification_report_imbalanced(mytest, myhattest))


default = metadf[metay_label].value_counts().argmax() # major class in training dataset
metadata = np.empty((0,metadf.shape[1]-1), np.float32)
metay = []
right = 0
count = 0
batch = 20


m_recommended = []
m_best = []

score_recommended = []
score_default = []

small_data = 5000
until_data = min(initial_data + small_data, int((df.shape[0]-omega)/gamma))

pbar = tqdm(range(initial_data, until_data))
for idx in pbar:
    train = df.iloc[idx*gamma:idx*gamma+omega]
    sel = df.iloc[idx*gamma+omega:(idx+1)*gamma+omega]

    xtrain, ytrain = train.drop(target, axis=1), train[target]
    xsel, ysel = sel.drop(target, axis=1), sel[target]

    mfe.fit(xtrain.values, ytrain.values)
    mfe_feats = [[v for k,v in zip(*mfe.extract()) if k not in\
                  missing_columns]]

    # predict best model
    yhat_model_name = np.argmax(metas.predict(mfe_feats))
    m_recommended.append(yhat_model_name)

    # metastrem score vs default score
    for model in models:
        model.fit(xtrain, ytrain)
    preds = [model.predict(xsel) for model in models]
    scores = [eval_metric(ysel, pred) for pred in preds]
    max_score = np.argmax(scores)

    score_recommended.append(scores[yhat_model_name])
    score_default.append(scores[default])

    metadata = np.append(metadata, mfe_feats, axis=0)
    metay.append(max_score)
    count += 1
    if yhat_model_name == max_score:
        right += 1
    pbar.set_description("Accuracy meta: {}".format(right/count))
    if count % batch == 0:
        metas = lgb.train(lgb_params,
                          init_model=metas,
                          train_set=lgb.Dataset(metadata[-batch:],
                                                metay[-batch:]))
    m_best.append(max_score)


print("Kappa: ", cohen_kappa_score(m_best, m_recommended))
print("GMean: ", geometric_mean_score(m_best, m_recommended))
print("Accuracy: ", accuracy_score(m_best, m_recommended))
print(classification_report(m_best, m_recommended))
print(classification_report_imbalanced(m_best, m_recommended))


print("Mean score Default {:.2f}+-{:.2f}".format(np.mean(score_default),
                                                 np.std(score_default)))
print("Mean score Recommended {:.2f}+-{:.2f}".\
      format(np.mean(score_recommended), np.std(score_recommended)))
