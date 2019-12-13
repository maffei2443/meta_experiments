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
from sklearn.model_selection import RandomizedSearchCV, LeaveOneOut
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
models = [
    SVC(gamma='scale'),
    RandomForestClassifier(random_state=42),
    GaussianNB()
]
lgb_params = {
    'boosting_type': 'dart',
    'learning_rate': 0.01,
    'tree_learner': 'feature',
    'metric': 'multi_error,multi_logloss',
    'objective': 'multiclassova',
    'num_class': len(models),
    # 'metric': 'binary_error,binary_logloss',
    # 'objective': 'binary',
    'is_unbalance': True,
    'verbose': -1,
    'seed': 42
}
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
    {}
]

def fine_tune(data, initial, gamma, omega, models, params, target, eval_metric):
    datasize = omega * initial // gamma - 1 # initial base data
    Xcv, ycv = data.loc[:datasize].drop([target], axis=1),\
        data.loc[:datasize, target]

    for model, param in tqdm(zip(models, params), total=len(models)):
        rscv = RandomizedSearchCV(model, param, random_state=42,
                                  scoring=make_scorer(metrics[eval_metric]),
                                  cv = gamma, n_jobs=-1)
        rscv.fit(Xcv, ycv)
        model.set_params(**rscv.best_params_)

def base_train(data, sup_mfe, unsup_mfe, gamma, omega, models, target, eval_metric):
    train = data.iloc[idx * gamma:idx * gamma + omega]
    sel = data.iloc[idx * gamma + omega:(idx+1) * gamma + omega]

    xtrain, ytrain = train.drop(target, axis=1), train[target]
    xsel, ysel = sel.drop(target, axis=1), sel[target]

    sup_mfe.fit(xtrain.values, ytrain.values)
    ft = sup_mfe.extract()
    sup_feats = {'sup_{}'.format(k):v for k,v in zip(*ft)}
    # unsup_mfe.fit(xsel.values)
    # ft = unsup_mfe.extract()
    # unsup_feats = {'unsup_{}'.format(k):v for k,v in zip(*ft)}

    # sup_feats.update(unsup_feats)
    mfe_feats = sup_feats

    for model in models:
        model.fit(xtrain, ytrain)
    preds = [model.predict(xsel) for model in models]
    scores = [metrics[eval_metric](ysel, pred) for pred in preds]
    max_score = np.argmax(scores)
    mfe_feats[metay_label] = max_score

    return mfe_feats

if __name__ == "__main__":
    ### LOAD DATA AND FINE TUNE TO INITIAL DATA
    df = pd.read_csv('data/elec2/electricity.csv')

    fine_tune(df, args.initial, args.gamma, args.omega, models, params,
              args.target, args.eval_metric)
    ### GENERATE METAFEATURES AND BEST CLASSIFIER FOR INITIAL DATA
    metadf = []
    sup_mfe = MFE(random_state=42)
    unsup_mfe = MFE(groups=['statistical'], random_state=42)
    for idx in tqdm(range(0, args.initial)):
        metadf.append(base_train(df, sup_mfe, unsup_mfe, args.gamma,
                                 args.omega, models, args.target,
                                 args.eval_metric))
    base_data = pd.DataFrame(metadf)
    base_data.to_csv('data/elec2/metabase.csv', index=False)
    base_data = pd.read_csv('data/elec2/metabase.csv')

    ### DROP MISSING DATA AND TRAIN METAMODEL
    missing_columns = base_data.columns[base_data.isnull().any()].values
    base_data.drop(columns=missing_columns, inplace=True)

    mX, mY = base_data.drop(metay_label, axis=1).values,\
        base_data[metay_label].values
    loo = LeaveOneOut()

    myhattest = []
    mytest = []
    for train_idx, test_idx in tqdm(loo.split(mX),
                                    total=args.initial):
        metas = lgb.train(lgb_params,
                      train_set=lgb.Dataset(mX[train_idx], mY[train_idx]))
        myhattest.append(np.argmax(metas.predict(mX[test_idx]), axis=1)[0])
        mytest.append(mY[test_idx][0])

    print("Kappa:    {:.2f}".format(cohen_kappa_score(mytest, myhattest)))
    print("GMean:    {:.2f}".format(geometric_mean_score(mytest, myhattest)))
    print("Accuracy: {:.2f}".format(accuracy_score(mytest, myhattest)))
    print(classification_report(mytest, myhattest))
    print(classification_report_imbalanced(mytest, myhattest))
    exit()

    ### ONLINE LEARNING
    metadf = []
    small_data = 5000000
    until_data = min(args.initial + small_data,
                     int((df.shape[0]-args.omega)/args.gamma))
    for idx in tqdm(range(args.initial, until_data)):
        metadf.append(base_train(df, sup_mfe, unsup_mfe, args.gamma,
                                 args.omega, models, args.target,
                                 args.eval_metric))
    online_data = pd.DataFrame(metadf)
    online_data.to_csv('data/elec2/metaonline.csv', index=False)
    print(online_data.head())
