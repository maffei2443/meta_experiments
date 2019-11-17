import numpy as np
import pandas as pd
import warnings

from scipy import stats
from pymfe.mfe import MFE
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression, SGDRegressor, Lasso
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import cohen_kappa_score, mean_squared_error, classification_report, accuracy_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced

from joblib import dump, load
import lightgbm as gbm

path = '../data/elec2/'

df = pd.read_csv(path + 'eletricity.csv')

ms_params = dict(
    window_size = 200,
    gamma_sel = 50,
    initial_data = 300, # Paper says 40 but too few for estimation
    target = 'class',
    eval_metric = accuracy_score,
    metay_label = 'clf')

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


datasize = ms_params['window_size'] *\
    ms_params['initial_data'] //\
    ms_params['gamma_sel'] - 1 # initial base data
Xcv, ycv = df.loc[:datasize].drop([ms_params['target']], axis=1),\
    df.loc[:datasize, ms_params['target']]

for model, param in tqdm(zip(models, params), total=len(models)):
    rscv = RandomizedSearchCV(model, param, n_jobs=-1, scoring=\
                              make_scorer(ms_params['eval_metric']))
    rscv.fit(Xcv, ycv)
    model.set_params(**rscv.best_params_)


metas = [RandomForestRegressor(random_state=42),
         RandomForestRegressor(random_state=42),
         RandomForestRegressor(random_state=42)]


mfe = MFE()
metadf = []

for idx in tqdm(range(ms_params['initial_data'])):
    train = df.iloc[idx * ms_params['gamma_sel']:idx *\
                    ms_params['gamma_sel'] + ms_params['window_size']]
    sel = df.iloc[idx * ms_params['gamma_sel'] +\
                  ms_params['window_size']:(idx+1) *\
                  ms_params['gamma_sel'] + ms_params['window_size']]

    xtrain, ytrain = train.drop(ms_params['target'], axis=1),\
        train[ms_params['target']]
    xsel, ysel = sel.drop(ms_params['target'], axis=1),\
        sel[ms_params['target']]

    mfe.fit(xtrain.values, ytrain.values)
    ft = mfe.extract()
    mfe_feats = dict(zip(*ft))

    for model in models:
        model.fit(xtrain, ytrain)
    preds = [model.predict(xsel) for model in models]
    scores = [ms_params['eval_metric'](ysel, pred) for pred in preds]

    for i, s in enumerate(scores):
        mfe_feats['score_clf'+str(i)] = s

    metadf.append(mfe_feats)


metadf = pd.DataFrame(metadf)
print(metadf.head())


ms_params['missing_columns'] = metadf.columns[metadf.isnull().any()].values
print(ms_params['missing_columns'])


metadf.dropna(axis=1, inplace=True) # Drop NaN row

metadf.to_csv(path + 'meta.csv', index=False)


dump(models, path + 'models.joblib')
dump(metas, path + 'metas.joblib')
dump(ms_params, path + 'ms_params.joblib');
