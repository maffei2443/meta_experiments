import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from joblib import dump, load

from pymfe.mfe import MFE
from pymfe.general import MFEGeneral
from tqdm import tqdm

from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem

from sklearn.metrics import cohen_kappa_score, mean_squared_error,\
    classification_report, accuracy_score, make_scorer, confusion_matrix

import lightgbm as lgb

import mlflow



def fine_tune(data, initial, gamma, omega, models, params, target, eval_metric):
    datasize = omega * initial // gamma - 1 # initial base data
    Xcv, ycv = data.loc[:datasize].drop([target], axis=1),\
        data.loc[:datasize, target]
    # input("divisao OK")
    for model, param in tqdm(zip(models, params), total=len(models)):
        
        try: 
            rscv = RandomizedSearchCV(model, param, random_state=42,
                                    scoring=make_scorer(metrics[eval_metric]),
                                    cv = gamma, n_jobs=-1)
            # print("RS ok...")
            try: 
                rscv.fit(Xcv, ycv)

            except Exception as e:
                print("huoou", e)
                input()
            model.set_params(**rscv.best_params_)
        except Exception as e:
            input(f"Escecao: {e}")

def base_train(models, data, idx, sup_mfe, gamma,
    omega, target, eval_metric):
    train = data.iloc[idx * gamma:idx * gamma + omega]
    sel = data.iloc[idx * gamma + omega:(idx+1) * gamma + omega]

    xtrain, ytrain = train.drop(target, axis=1), train[target]
    xsel, ysel = sel.drop(target, axis=1), sel[target]

    sup_mfe.fit(xtrain.values, ytrain.values)
    ft = sup_mfe.extract()
    mfe_feats = dict(zip(*ft))

    for model in models:
        model.fit(xtrain, ytrain)
    preds = [model.predict(xsel) for model in models]
    scores = [metrics[eval_metric](ysel, pred) for pred in preds]
    # print(scores)
    # input()
    # raise BaseException("uuuu")
    return mfe_feats, scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process params for metastream.')
    parser.add_argument('--omega', type=int, help='train window size.')
    parser.add_argument('--gamma', type=int, help='selection window size.')
    parser.add_argument('--initial', type=int, help='initial data.')
    parser.add_argument('--target', help='target label.')
    parser.add_argument('--eval_metric', help='eval metric for metastream.')
    parser.add_argument('--path', help='data path.')
    parser.add_argument('--metay', help='metay label.', default='best_classifier')
    parser.add_argument('--test_size_ts', help='test_size_ts.', default=100)
    parser.add_argument('--quick', help='wheter to use only general mtf.', 
        default=1)

    args, _ = parser.parse_known_args()

    metay_label = args.metay
    test_size_ts = args.test_size_ts
    args.initial += test_size_ts

    models = [
        # misc
        Pipeline([("nys", Nystroem(random_state=42)),
                ("svm", LinearSVC(dual=False))]),
        RandomForestClassifier(random_state=42),
        # bayesian models
        GaussianNB(),
        ComplementNB(),
        # linear models
        # LogisticRegression(random_state=42, n_jobs=-1),

    ]
    params = [
        dict(svm__C=[1,10,100],        
            nys__kernel=['poly', 'rbf', 'sigmoid'],
        ),
        dict(max_depth=[3, 5, None],
            n_estimators=[10, 200, 300],
            min_samples_split=stats.randint(2, 11),
            criterion=["gini", "entropy"],
            bootstrap=[True, False],
        ),
        dict(
            var_smoothing=[1e-09]
        ),
        dict(
            alpha=[0., 1.],
            norm=[True, False],
        ),
        # dict(
        #     max_iter=[100, 150],
        #     multi_class=['ovr', 'multinomial'],
        # )
    ]
    
    if args.quick:
        params = []


    p_dict = dict(
        svm_nys=dict(svm__C=[1,10,100],        
            nys__kernel=['poly', 'rbf', 'sigmoid'],
        ),
        rf=dict(max_depth=[3, 5, None],
            n_estimators=[10, 200, 300],
            min_samples_split=stats.randint(2, 11),
            criterion=["gini", "entropy"],
            bootstrap=[True, False],
        ),
        gaussian_nb=dict(
            var_smoothing=[1e-09]
        ),
        complement_nb=dict(
            alpha=[0., 1.],
            norm=[True, False],
        ),
    )

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




    path = args.path
    run = mlflow.start_run()

    print("RUN:", run.info.run_id)
    ### LOAD DATA AND FINE TUNE TO INITIAL DATA
    print("[FINETUNING BASE MODELS]")
    df = pd.read_csv(path +'data.csv')
    from sklearn.preprocessing import LabelEncoder as LE
    df[args.target] = LE().fit_transform(df[args.target])
    print('target:\n\t', df[args.target])
    # print("skipping fine tune...")

    fine_tune(df, args.initial, args.gamma, args.omega, models, params,
              args.target, args.eval_metric)

    dump(models, path + 'models.joblib')
    models = load(path + 'models.joblib')


    ### GENERATE METAFEATURES AND BEST CLASSIFIER FOR INITIAL DATA
    print("[GENERATE METAFEATURE]")
    metadf = []
    if args.quick:
        print("gotta go fassst")
        sup_mfe = MFE(
            groups=['general'], features=['sd', 'min', 'max'], 
            num_cv_folds=1, summary=('mean',))
        
        # base_data = pd.read_csv(path + 'metabase.csv')
    else:
        print("normal flow...")
        sup_mfe = MFE(features=["best_node","elite_nn","linear_discr",
                                "naive_bayes","one_nn","random_node","worst_node",
                                "can_cor","cor", "cov","g_mean",
                                "gravity","h_mean","iq_range","kurtosis",
                                "lh_trace","mad","max","mean","median","min",
                                "nr_cor_attr","nr_disc","nr_norm","nr_outliers",
                                "p_trace","range","roy_root","sd","sd_ratio",
                                "skewness","sparsity","t_mean","var","w_lambda"],
                    random_state=42)
    # unsup_mfe = MFE(groups=["statistical"], random_state=42)
    off_scores = []

    # This loop generates the data composing the
    # initial data, which means the initial metabase

    # Se nao eh modo rapido OU [eh modo rapido e] nao tem metabase criada, crie-a
    if not args.quick or not os.path.isfile(path + 'metabase_quick.csv'):
        for idx in tqdm(range(0, args.initial)):
            mfe_feats, scores = base_train(models, df, idx, sup_mfe, args.gamma,
                                        args.omega, args.target,
                                        args.eval_metric)
            off_scores.append(scores)
            # max_score = np.argmax(scores)
            mfe_feats[metay_label] = np.argmax(scores)
            metadf.append(mfe_feats)
            dump(off_scores, path + 'off_scores.joblib')
        base_data = pd.DataFrame(metadf)
        if args.quick:
            base_data.to_csv(path + 'metabase_quick.csv', index=False)
        else:
            base_data.to_csv(path + 'metabase.csv', index=False)
    else:        
        print("READING CACHED METABASE...")
        base_data = pd.read_csv(path + 'metabase_quick.csv')
    
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

    kappas = []
    gmeans = []
    accurs = []
    off_targets = []
    off_preds = []
    preds_lis = []
    for i in tqdm(range(test_size_ts)):
        itest = i+args.omega
        metas = lgb.train(lgb_params,
                          train_set=lgb.Dataset(mX[i:i+itest],
                                                mY[i:i+itest]))


        raw_preds=metas.predict(mX[itest:itest+args.gamma])
        # print("type_raw_pred:", type(raw_preds))
        # input()
        # print("raw_preds:", raw_preds)
        # input('next...')
        # preds = np.argmax(raw_preds > .5, 1, 0)
        preds = np.apply_along_axis(np.argmax, 1, raw_preds)   
        preds_lis.append(preds)
        # print("preds:", preds)
        targets = mY[itest:itest+args.gamma]
        # print("targets:", targets)
        if np.array_equal(preds, targets):
            kappas.append(1.0)
        else:
            kappas.append(cohen_kappa_score(targets, preds))
        mlflow.log_metric('off_kappa', kappas[-1], i)

        gmeans.append(geometric_mean_score(targets, preds))
        mlflow.log_metric('off_gmean', gmeans[-1], i)

        accurs.append(accuracy_score(targets, preds))
        mlflow.log_metric('off_accur', accurs[-1], i)


        off_targets.append(targets)
        off_preds.append(preds)
    df_preds = pd.DataFrame(preds_lis)
    df_preds.to_csv(path + 'meta_preds.csv')

    dump(off_preds, path + 'off_preds.joblib')
    dump(off_targets, path + 'off_targets.joblib')
    metas.save_model(path + 'metamodel.txt')
    metas = lgb.Booster(model_file=path + 'metamodel.txt')
    print("Kappa:    {:.3f}+-{:.3f}".format(np.mean(kappas), np.std(kappas)))
    print("GMean:    {:.3f}+-{:.3f}".format(np.mean(gmeans), np.std(gmeans)))
    print("Accuracy: {:.3f}+-{:.3f}".format(np.mean(accurs), np.std(accurs)))

    mlflow.log_metric('off_kappa_mean', np.mean(kappas))
    mlflow.log_metric('off_gmean_mean', np.mean(gmeans))
    mlflow.log_metric('off_accur_mean', np.mean(accurs))

    mlflow.log_metric('off_kappa_std', np.std(kappas))
    mlflow.log_metric('off_gmean_std', np.std(gmeans))
    mlflow.log_metric('off_accur_std', np.std(accurs))


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
    f_importance = []

    score_recommended = []
    score_default = []
    score_svm = []
    score_rf = []

    small_data = 5000000
    until_data = min(args.initial + small_data,
                     int((df.shape[0]-args.omega)/args.gamma))
    # for idx in tqdm(range(args.initial, until_data)):
    for idx in tqdm(range(args.initial, args.initial + 100)):
        mfe_feats, scores = base_train(models, df, idx, sup_mfe, args.gamma,
                                       args.omega, args.target,
                                       args.eval_metric)
        mfe_feats = [[mfe_feats[k] for k in mfe_feats if k\
                     not in missing_columns]]
        metadf = np.append(metadf, mfe_feats, axis=0)
        max_score = np.argmax(scores)
        meta_pred = metas.predict(mfe_feats)
        # print('meta_pred:', meta_pred)
        recommended = np.argmax(meta_pred[0])
        # input()
        # recommended = np.where(meta_pred>.5, 1, 0)[0]

        score_svm.append(scores[0])
        score_rf.append(scores[1])
        m_recommended.append(recommended)
        m_best.append(max_score)
        # print("default:", default)
        # print("recommended:", recommended)
        m_diff.append(scores[recommended] - scores[default])
        score_recommended.append(scores[recommended])
        score_default.append(scores[default])

        metay.append(max_score)
        # Quando atinge um múltiplo de gamma (tamanho da janela de seleção)
        # há que se treinar um novo modelo
        counter += 1
        if counter % args.gamma == 0:
            metas = lgb.train(lgb_params,
                              train_set=lgb.Dataset(metadf[-args.omega:],
                                                    metay[-args.omega:]))
            f_importance.append(metas.feature_importance())

    dump(m_recommended, path + 'recommended.joblib')
    dump(m_best, path + 'best.joblib')
    dump(m_diff, path + 'difference.joblib')
    dump(score_recommended, path + 'score_reco.joblib')
    dump(score_default, path + 'score_def.joblib')
    dump(score_svm, path + 'score_svm.joblib')
    dump(score_rf, path + 'score_rf.joblib')
    dump(metadf, path + 'metadf.joblib')
    dump(f_importance, path + 'tfi.joblib')
    print("Kappa: ", cohen_kappa_score(m_best, m_recommended))
    print("GMean: ", geometric_mean_score(m_best, m_recommended))
    print("Accuracy: ", accuracy_score(m_best, m_recommended))
    print(classification_report(m_best, m_recommended))
    print(classification_report_imbalanced(m_best, m_recommended))
