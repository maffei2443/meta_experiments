import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use('ggplot')

import warnings
warnings.filterwarnings("ignore")

from pymfe.mfe import MFE
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, mean_squared_error, classification_report, accuracy_score, make_scorer
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced

from joblib import dump, load
import lightgbm as lgb

path = '../data/elec2/'

models = load(path + 'models.joblib')
metas = load(path + 'metas.joblib')
ms_params = load(path + 'ms_params.joblib')


df = pd.read_csv(path + 'eletricity.csv')
df.head()


metadf = pd.read_csv(path + 'meta.csv')
metadf.head()


filtered_columns = metadf.columns.values


print(filtered_columns)


metadf.iloc[:,-len(models):].idxmax(axis=1).value_counts()/metadf.shape[0]

mxtrain, mxtest, mytrain, mytest = train_test_split(metadf.\
         iloc[:,:-len(models)], metadf.iloc[:,-len(models):], random_state=42)


myhattest = []
for i, meta in enumerate(metas):
    meta.fit(mxtrain, mytrain.iloc[:,i])
    myhattest.append(meta.predict(mxtest))
myhattest = pd.DataFrame(myhattest).T.idxmax(axis=1).values
myreal = mytest.idxmax(axis=1).str[-1].astype(int).values
print("Kappa: ", cohen_kappa_score(myreal, myhattest))
print("GMean: ", geometric_mean_score(myreal, myhattest))
print("Accuracy: ", accuracy_score(myreal, myhattest))
print(classification_report_imbalanced(myreal, myhattest))


default = int(metadf.iloc[:,-len(models):].idxmax(axis=1).value_counts()\
              .index[0][-1]) # major class in training dataset
metadata = []
metay = []
count = 0
batch = 20


m_recommended = []
m_best = []

score_recommended = []
score_default = []

mfe = MFE()

small_data = 1000
until_data = min(ms_params['initial_data'] + small_data,
                 int((df.shape[0]-ms_params['window_size']) /\
                     ms_params['gamma_sel']))

print("Running from {} to {}.".format(ms_params['initial_data'], until_data))

pbar = tqdm(range(ms_params['initial_data'], until_data))
for idx in pbar:
    train = df.iloc[idx*ms_params['gamma_sel']:idx*ms_params['gamma_sel'] +
                    ms_params['window_size']]
    sel = df.iloc[idx * ms_params['gamma_sel'] + ms_params['window_size']:\
                  (idx+1) * ms_params['gamma_sel'] + ms_params['window_size']]

    xtrain, ytrain = train.drop(ms_params['target'], axis=1),\
        train[ms_params['target']]
    xsel, ysel = sel.drop(ms_params['target'], axis=1),\
        sel[ms_params['target']]

    mfe.fit(xtrain.values, ytrain.values)
    mfe_feats = [[v for k,v in zip(*mfe.extract()) if k not in\
                  ms_params['missing_columns']]]

    # predict best model
    for model in models:
        model.fit(xtrain, ytrain)
    preds = [model.predict(xsel) for model in models]
    scores = [ms_params['eval_metric'](ysel, pred) for pred in preds]
    max_score = np.argmax(scores)

    yhat_model_name = np.argmax([mts.predict(mfe_feats) for mts in metas])
    metas = [mts.train(ms_params['meta_params'],
                       init_model=metas[i],
                       train_set=lgb.Dataset(metadata[-batch:],
                                             metay[-batch:][i])
                       ) for i, mts in enumerate(metas)]
    m_recommended.append(yhat_model_name)
    m_best.append(max_score)

    # metastrem score vs default score
    pbar.set_description("Accuracy meta: {}".format(scores[yhat_model_name]))
    score_recommended.append(scores[yhat_model_name])
    score_default.append(scores[default])
    metadata.append(mfe_feats[0])
    metay.append(scores)


print("Kappa: ", cohen_kappa_score(m_best, m_recommended))
print("GMean: ", geometric_mean_score(m_best, m_recommended))
print("Accuracy: ", accuracy_score(m_best, m_recommended))
print(classification_report(m_best, m_recommended))
print(classification_report_imbalanced(m_best, m_recommended))


print("Mean score Default {:.2f}+-{:.2f}".format(np.mean(score_default),\
                                                 np.std(score_default)))
print("Mean score Recom. {:.2f}+-{:.2f}".format(np.mean(score_recommended),\
                                                np.std(score_recommended)))
