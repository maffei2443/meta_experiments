import argparse
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

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
parser.add_argument('--omega', help='train window size.')
parser.add_argument('--gamma', help='selection window size.')
parser.add_argument('--initial', help='initial data.')
parser.add_argument('--target', help='target label.')
parser.add_argument('--eval_metric', help='eval metric for metastream.')
parser.add_argument('--metay', help='')

args = parser.parse_args()

metadf = []

metrics = {
    'acc': accuracy_score
}
metay_label = 'clf'


params = [
        {"C": [1,10,100],
         "kernel": ["rbf", "linear", "poly", "sigmoid"]},
        {"max_depth": [3, None],
         "max_features": stats.randint(1, 9),
         "min_samples_split": stats.randint(2, 11),
         "bootstrap": [True, False],
         "criterion": ["gini", "entropy"]},
        {}]

models = [
        SVC(),
        RandomForestClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42)]

omega = args.omega
gamma = args.gamma
initial_data = args.initial
target = args.target
eval_metric = metrics[args.eval_metric]

datasize = omega * initial_data // gamma - 1 # initial base data
Xcv, ycv = df.loc[:datasize].drop([target], axis=1), df.loc[:datasize, target]

for model, param in tqdm(zip(models, params), total=len(models)):
    rscv = RandomizedSearchCV(model, param, n_jobs=-1,
                              scoring=make_scorer(eval_metric))
    rscv.fit(Xcv, ycv)
    model.set_params(**rscv.best_params_)

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


metas = lgbm.train(lgbm_params,
                   train_set=lgbm.Dataset(mxtrain, mytrain),
                   valid_set=lgbm.Dataset(mxtest, mytest),
                   early_stopping_rounds=20)

myhattest = metas.predict(mxtest)
print("Kappa: ", cohen_kappa_score(mytest, myhattest))
print("GMean: ", geometric_mean_score(mytest, myhattest))
print("Accuracy: ", accuracy_score(mytest, myhattest))
print(classification_report(mytest, myhattest))
print(classification_report_imbalanced(mytest, myhattest))


default = metadf[metay_label].value_counts().argmax() # major class in training dataset
metadata = np.empty((0,21), np.float64)
metay = []
count = 0
batch = 20


m_recommended = []
m_best = []

score_recommended = []
score_default = []

small_data = 5000
until_data = min(initial_data + small_data, int((df.shape[0]-omega)/gamma))

exit(0)

with localconverter(ro.default_converter + pandas2ri.converter):
    for idx in tqdm_notebook(range(initial_data, until_data)):
        train = df.iloc[idx*gamma:idx*gamma+omega]
        sel = df.iloc[idx*gamma+omega:(idx+1)*gamma+omega]

        xtrain, ytrain = train.drop(target, axis=1), train[target]
        xsel, ysel = sel.drop(target, axis=1), sel[target]

        mfe_feats = []
        ecol = importr("ECoL")
        mfe_feats = ecol.complexity(xtrain, ytrain)
        mfe_feats = np.delete(mfe_feats, 8).reshape(1, -1)

        # predict best model
        yhat_model_name = int(metas.predict(mfe_feats)[0])
        m_recommended.append(yhat_model_name)

        # metastrem score vs default score
        score1 = eval_metric(ysel, metas._learners[yhat_model_name].fit(xtrain, ytrain).predict(xsel))
        score_recommended.append(score1)
        if default != yhat_model_name:
            score2 = eval_metric(ysel, metas._learners[default].fit(xtrain, ytrain).predict(xsel))
        else:
            score2 = score1
        score_default.append(score2)

        metas.base_fit(xtrain, ytrain)
        preds = metas.base_predict(xsel)
        scores = [eval_metric(ysel, pred) for pred in preds]
        max_score = np.argmax(scores)

        metadata = np.append(metadata, mfe_feats, axis=0)
        metay.append([max_score])
        count += 1
        if count % batch == 0:
            metas._metalearner.partial_fit(metadata[-batch:], metay[-batch:])

        m_best.append(max_score)


# In[ ]:


print("Kappa: ", cohen_kappa_score(m_best, m_recommended))
print("GMean: ", geometric_mean_score(m_best, m_recommended))
print("Accuracy: ", accuracy_score(m_best, m_recommended))
print(classification_report(m_best, m_recommended))
print(classification_report_imbalanced(m_best, m_recommended))


# In[ ]:


plt.hist(score_default, alpha=.5)
plt.hist(score_recommended, alpha=.5)
plt.ylabel('Count')
print("Mean score Default {:.2f}+-{:.2f}".format(np.mean(score_default), np.std(score_default)))
print("Mean score Recommended {:.2f}+-{:.2f}".format(np.mean(score_recommended), np.std(score_recommended)))


# In[ ]:


plt.scatter(score_default, score_recommended, alpha=.25)
plt.xlabel('Default')
plt.ylabel('Recommended');


# In[ ]:


timedf = pd.DataFrame(metadata)


# In[ ]:


timedf.head()


# In[ ]:


window_score = min(500, small_data)
for i in range(21):
    fig, ax1 = plt.subplots(figsize=(15,2))
    ax1.plot(score_default[-window_score:], color='C0')
    ax1.set_ylabel("Score", color='C0')
    ax1.set_xlabel("Pseudo-time")
    ax2 = ax1.twinx()
    ax2.plot(timedf[i].values[-window_score:], color='C1')
    ax2.set_ylabel("Metafeature {}".format(i), color='C1')
    plt.title("Score variation from default algorithm compared to metafeature");


# In[ ]:


fig, ax1 = plt.subplots(figsize=(15,2))
ax1.plot(score_recommended[-window_score:])
ax1.set_ylabel("Score")
ax1.set_xlabel("Pseudo-time")
plt.title("Score over time");


# In[ ]:


N = 10
moving_aves = np.convolve(score_recommended[-window_score:], np.ones((N,))/N, mode='valid')


# In[ ]:


fig, ax1 = plt.subplots(figsize=(15,2))
ax1.plot(moving_aves)
ax1.set_ylabel("Score")
ax1.set_xlabel("Pseudo-time")
plt.title("Moving average over time");


# In[ ]:


window_score = min(500, small_data)
N = 10
moving_avg = np.convolve(score_default[-window_score:], np.ones((N,))/N, mode='valid')
for i in range(21):
    fig, ax1 = plt.subplots(figsize=(15,2))
    ax1.plot(moving_avg, color='C0')
    ax1.set_ylabel("Score", color='C0')
    ax1.set_xlabel("Pseudo-time")
    ax2 = ax1.twinx()
    ax2.plot(timedf[i].values[-window_score:], color='C1')
    ax2.set_ylabel("Metafeature {}".format(i), color='C1')
    plt.title("Score variation from default algorithm compared to metafeature");

