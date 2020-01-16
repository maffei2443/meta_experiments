import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from joblib import dump, load

from pymfe.mfe import MFE
from tqdm import tqdm

from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score, mean_squared_error,\
    classification_report, accuracy_score, make_scorer, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, LeaveOneOut
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced

import lightgbm as lgb



parser = argparse.ArgumentParser(description='Process params for metastream.')
parser.add_argument('--omega', type=int, help='train window size.')
parser.add_argument('--gamma', type=int, help='selection window size.')
parser.add_argument('--initial', type=int, help='initial data.')
parser.add_argument('--target', help='target label.')
parser.add_argument('--eval_metric', help='eval metric for metastream.')
parser.add_argument('--path', help='data path.')
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
    'learning_rate': 0.1,
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
    "kernel": ["rbf", "poly"]},
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

def base_train(data, idx, sup_mfe, unsup_mfe, gamma, omega, models, target, eval_metric):
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

    return mfe_feats, scores

if __name__ == "__main__":
    path = args.path
    ### LOAD DATA AND FINE TUNE TO INITIAL DATA
    print("[FINETUNING BASE MODELS]")
    df = pd.read_csv(path +'data.csv')

    fine_tune(df, args.initial, args.gamma, args.omega, models, params,
              args.target, args.eval_metric)
    dump(models, path + 'models.joblib')
    models = load(path + 'models.joblib')
    ### GENERATE METAFEATURES AND BEST CLASSIFIER FOR INITIAL DATA
    print("[GENERATE METAFEATURE]")
    metadf = []
    sup_mfe = MFE(random_state=42)
    unsup_mfe = MFE(groups=['statistical'], random_state=42)
    for idx in tqdm(range(0, args.initial)):
        mfe_feats, scores = base_train(df, idx, sup_mfe, unsup_mfe, args.gamma,
                                       args.omega, models, args.target,
                                       args.eval_metric)
        max_score = np.argmax(scores)
        mfe_feats[metay_label] = max_score
        metadf.append(mfe_feats)
    base_data = pd.DataFrame(metadf)
    base_data.to_csv(path + 'metabase.csv', index=False)
    base_data = pd.read_csv(path + 'metabase.csv')
    print("Frequency statistics in metabase:")
    for idx, count in base_data[metay_label].value_counts().items():
        print("\t{:25}{:.3f}".format(str(models[idx]).split('(')[0],
                                     count/args.initial))

    ### DROP MISSING DATA AND TRAIN METAMODEL
    print("[OFFLINE LEARNING]")
    missing_columns = base_data.columns[base_data.isnull().any()].values
    base_data.drop(columns=missing_columns, inplace=True)

    mX, mY = base_data.drop(metay_label, axis=1).values,\
        base_data[metay_label].values
    loo = LeaveOneOut()

    myhattest = []
    mytest = []
    for train_idx, test_idx in tqdm(loo.split(mX), total=args.initial):
        metas = lgb.train(lgb_params, train_set=lgb.Dataset(mX[train_idx],
                                                            mY[train_idx]))
        myhattest.append(np.argmax(metas.predict(mX[test_idx]), axis=1)[0])
        mytest.append(mY[test_idx][0])

    metas.save_model(path + 'metamodel.txt')
    metas = lgb.Booster(model_file=path + 'metamodel.txt')
    print("Kappa:    {:.3f}".format(cohen_kappa_score(mytest, myhattest)))
    print("GMean:    {:.3f}".format(geometric_mean_score(mytest, myhattest)))
    print("Accuracy: {:.3f}".format(accuracy_score(mytest, myhattest)))
    print(confusion_matrix(mytest, myhattest))
    print(classification_report(mytest, myhattest))
    print(classification_report_imbalanced(mytest, myhattest))
    importance = metas.feature_importance()
    fnames = base_data.columns
    dump(importance, path + 'importance.joblib')
    dump(fnames, path + 'fnames.joblib')

    ### ONLINE LEARNING
    print("[ONLINE LEARNING]")
    default = base_data[metay_label].value_counts().argmax()
    metadf = np.empty((0, base_data.shape[1]-1), np.float32)
    metay = []
    counter = 0

    m_recommended = []
    m_best = []
    m_diff = []

    score_recommended = []
    score_default = []

    small_data = 5000000
    until_data = min(args.initial + small_data,
                     int((df.shape[0]-args.omega)/args.gamma))
    for idx in tqdm(range(args.initial, until_data)):
        mfe_feats, scores = base_train(df, idx, sup_mfe, unsup_mfe, args.gamma,
                                       args.omega, models, args.target,
                                       args.eval_metric)
        mfe_feats = [[mfe_feats[k] for k in mfe_feats if k\
                     not in missing_columns]]
        metadf = np.append(metadf, mfe_feats, axis=0)
        max_score = np.argmax(scores)
        recommended = np.argmax(metas.predict(mfe_feats))

        m_recommended.append(recommended)
        m_best.append(max_score)
        m_diff.append(scores[recommended] - scores[default])

        metay.append(max_score)
        counter += 1
        if counter % args.gamma == 0:
            metas = lgb.train(lgb_params, init_model=metas,
                              train_set=lgb.Dataset(metadf[-args.gamma:],
                                                    metay[-args.gamma:]))

    dump(m_diff, path + 'difference.joblib')
    dump(m_best, path + 'best.joblib')
    dump(m_recommended, path + 'recommended.joblib')
    dump(metadf, path + 'metadata.joblib')
    print("Kappa: ", cohen_kappa_score(m_best, m_recommended))
    print("GMean: ", geometric_mean_score(m_best, m_recommended))
    print("Accuracy: ", accuracy_score(m_best, m_recommended))
    print(classification_report(m_best, m_recommended))
    print(classification_report_imbalanced(m_best, m_recommended))
