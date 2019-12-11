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
from sklearn.naive_bayes import GaussianNB
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
metay_label = 'best_classifier'

def fine_tune(data, initial, gamma, omega, models, params, target, eval_metric):
    datasize = omega * initial // gamma - 1 # initial base data
    Xcv, ycv = data.loc[:datasize].drop([target], axis=1),\
        data.loc[:datasize, target]

    for model, param in tqdm(zip(models, params), total=len(models)):
        rscv = RandomizedSearchCV(model, param, n_jobs=-1,
                              scoring=make_scorer(metrics[eval_metric]))
        rscv.fit(Xcv, ycv)
        model.set_params(**rscv.best_params_)

def base_train(data, start, stop, gamma, omega, models, target, eval_metric):
    eval_metric = metrics[eval_metric]
    metadf = []
    sup_mfe = MFE(groups=['statistical', 'complexity'], random_state=42)
    # unsup_mfe = MFE(groups=['statistical'], random_state=42)
    for idx in tqdm(range(start, stop)):
        train = data.iloc[idx * gamma:idx * gamma + omega]
        sel = data.iloc[idx * gamma + omega:(idx+1) * gamma + omega]

        xtrain, ytrain = train.drop(target, axis=1), train[target]
        xsel, ysel = sel.drop(target, axis=1), sel[target]

        sup_mfe.fit(xtrain.values, ytrain.values)
        ft = sup_mfe.extract()
        sup_feats = {'sup_{}'.format(k):v for k,v in zip(*ft)}
        # unsup_mfe.fit(xsel.values)
        # ft = unsup_mfe.extract()
        # unsup_feats = dict(('unsup_{}'.format(k), v) for k,v in zip(*ft))

        # sup_feats.update(unsup_feats)
        mfe_feats = sup_feats

        for model in models:
            model.fit(xtrain, ytrain)
        preds = [model.predict(xsel) for model in models]
        scores = [eval_metric(ysel, pred) for pred in preds]
        max_score = np.argmax(scores)
        mfe_feats[metay_label] = max_score

        metadf.append(mfe_feats)

    return pd.DataFrame(metadf)

if __name__ == "__main__":
    ### SPECIFY PARAMETERS FOR BASE AND META MODELS
    models = [
        SVC(gamma='scale'),
        RandomForestClassifier(random_state=42)
    ]
    metrics = {
        'acc': accuracy_score
    }
    params = [
            {"C": [1,10,100],
             "kernel": ["rbf", "linear", "poly", "sigmoid"]},
            {"max_depth": [3, None],
             "n_estimators": [100, 200, 300, 500],
             "max_features": stats.randint(1, 9),
             "min_samples_split": stats.randint(2, 11),
             "bootstrap": [True, False],
             "criterion": ["gini", "entropy"]},
    ]
    lgb_params = {
        'boosting_type': 'dart',
        'learning_rate': 0.01,
        'tree_learner': 'feature',
        'metric': 'multi_error,multi_logloss',
        'objective': 'multiclass',
        'num_class': 3,
        # 'metric': 'binary_error,binary_logloss',
        # 'objective': 'binary',
        'is_unbalance': True,
        'seed': 42
    }
    ### LOAD DATA AND FINE TUNE TO INITIAL DATA
    df = pd.read_csv('data/elec2/electricity.csv')

    fine_tune(df, args.initial, args.gamma, args.omega, models, params,
              args.target, args.eval_metric)
    ### GENERATE METAFEATURES AND BEST CLASSIFIER FOR INITIAL DATA
    base_data = base_train(df, 0, args.initial, args.gamma, args.omega,
                           models, args.target, args.eval_metric)
    base_data.to_csv('data/elec2/metabase.csv', index=False)
    base_data = pd.read_csv('data/elec2/metabase.csv')

    ### DROP MISSING DATA AND TRAIN METAMODEL
    missing_columns = base_data.columns[base_data.isnull().any()].values
    base_data.drop(columns=missing_columns, inplace=True)

    mxtrain, mxtest, mytrain, mytest =\
        train_test_split(base_data.drop(metay_label, axis=1),
                         base_data[metay_label], random_state=42)

    metas = lgb.train(lgb_params, train_set=lgb.Dataset(mxtrain, mytrain),
                      valid_sets=lgb.Dataset(mxtest, mytest),
                      num_boost_round=500,
                      early_stopping_rounds=10)

    myhattest = np.argmax(metas.predict(mxtest), axis=1)
    print("Kappa: ", cohen_kappa_score(mytest, myhattest))
    print("GMean: ", geometric_mean_score(mytest, myhattest))
    print("Accuracy: ", accuracy_score(mytest, myhattest))
    print(classification_report(mytest, myhattest))
    print(classification_report_imbalanced(mytest, myhattest))
    exit()

    ### ONLINE LEARN
    small_data = 5000000
    until_data = min(args.initial + small_data,
                     int((df.shape[0]-args.omega)/args.gamma))

    online_data = base_train(df, args.initial, until_data, args.gamma,
                             args.omega, models, args.target, args.eval_metric)
    online_data.to_csv('data/elec2/metaonline.csv', index=False)
    print(online_data.head())
