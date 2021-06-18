import os
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import make_scorer, confusion_matrix

import lightgbm as lgb
import mlflow

from collections import namedtuple                                      
NMP = namedtuple('NameModelParam', 'name model params')

def fine_tune(data, data_temporal, initial, gamma, omega, m_p, 
  target, eval_metric):
  datasize = omega * initial // gamma - 1 # initial base data
  # tmp =
  Xcv, ycv = data.loc[:datasize].drop([target], axis=1),\
    data.loc[:datasize, target]
  for tup in tqdm(m_p, total=len(m_p)):
      
      print("Finetuning %s" % tup.name , "..." )                  
      # rscv = RandomizedSearchCV(tup.model, tup.params, random_state=42,
      #             scoring=make_scorer(metrics[eval_metric]),
      #             cv = gamma, n_jobs=-1)
      rscv = GridSearchCV(tup.model, tup.params,
                  # random_state=42,
                  scoring=make_scorer(metrics[eval_metric]),
                  cv = gamma, n_jobs=-1)
      rscv.fit(Xcv, ycv)

      tup.model.set_params(**rscv.best_params_)

def train_sel_split(df, left, mid, right, target):
  train, test = df.iloc[left:mid], df.iloc[mid:right]
  xtrain, ytrain = train.drop(target, axis=1), train[target]
  xtest, ytest = test.drop(target, axis=1), test[target]

  return xtrain, xtest, ytrain, ytest


def base_train_test(m_p, data, idx, sup_mfe, gamma,
  omega, target, eval_metric, unsup_mfe=None):

  left = idx * gamma
  mid = left + omega
  right = mid + gamma
  # print('l, m, r: {}, {}, {}'.format(left, mid, right) )
  xtrain, xsel, ytrain, ysel = train_sel_split(
    data, left, mid, right, target,
  )

  for tup in m_p:
    tup.model.fit(xtrain, ytrain)
  preds = []
  for tup in m_p:
    try:
      preds.append(tup.model.predict(xsel))
    except Exception as e:
      print(tup.name)
      input(e)
      raise e
    
  preds = [tup.model.predict(xsel) for tup in m_p]
  # print("predict ok")
  scores = [metrics[eval_metric](ysel, pred) for pred in preds]
  score_dict = {
    tup.name: metrics[eval_metric](ysel, tup.model.predict(xsel))
    for tup in m_p}
  
  sup_mfe.fit(xtrain.values, ytrain.values)
  mfe_feats = {}
  if unsup_mfe:
    unsup_mfe.fit(xtrain.values)
    unsup_feats = {f'unsup_{k}':v for (k,v) in zip(*unsup_mfe.extract())}
    mfe_feats.update(unsup_feats)

  sup_ft = sup_mfe.extract()
  mfe_feats.update( dict(zip(*sup_ft)) )

  return mfe_feats, scores, score_dict


if __name__ == "__main__":
  run = mlflow.start_run()

  parser = argparse.ArgumentParser(description='Process params for metastream.')
  parser.add_argument('--omega', type=int, help='train window size.')
  parser.add_argument('--gamma', type=int, help='selection window size.')
  parser.add_argument('--initial', type=int, help='initial data.')
  parser.add_argument('--target', help='target label.')
  parser.add_argument('--eval_metric', help='eval metric for metastream.')
  parser.add_argument('--path', help='data path.')
  parser.add_argument('--metay', help='metay label.', default='best_classifier')
  parser.add_argument('--test_size_ts', help='test_size_ts.', default=100, type=int)
  parser.add_argument('--fine_tune', help='ignores parameter tunning.', 
    default=0, type=int)
  parser.add_argument('--quick', help='whether to apply certain tricks just to finish faster.', 
    default=0, type=int)
  parser.add_argument('--cache', help='wheter to use only general mtf.', 
    default=0, type=int)
  
  parser.add_argument('--save_metamodel', help='wheter to use only general mtf.', 
    default=1, type=int)
  parser.add_argument('--temporal_feats', help='wheter to use temporal feature extraction.', 
    default=0, type=int)
  parser.add_argument('--unsup_features', help='wheter to use unsupervised metafeatures.', 
    default=1, type=int)
  parser.add_argument('--temporal_features', help='wheter to use temporal feature extraction.', 
    default=1, type=int)
  
  
  args, _ = parser.parse_known_args()
  mlflow.log_params({
    'cli_'+k: v for (k,v) in args.__dict__.items()
  })
  if not args.fine_tune and not args.cache:
    print("WHOLE EXPERIMENT....")
  if args.cache:
    print("GONNA USE CACHE!")
    # input()
  metay_label = args.metay
  test_size_ts = args.test_size_ts
  args.initial += test_size_ts


  m_p = [

    NMP(
      name='linear_svm',
      model=LinearSVC(dual=False, random_state=42),
      params=dict(C=[1,10, 50,100],    
                  tol=[.0001, .001, ],
                  max_iter=[1000, 1500],
                  )
    ),
    NMP(
      name='decision_tree',
      model=DecisionTreeClassifier(random_state=42),
      params=dict(criterion=['gini', 'entropy'],    
                  max_depth=[2, 3, 5, 8, 10],
                  max_features=[0.7, 'sqrt', 'log2'],
                  ccp_alpha=[.0, .3, .7],)
    ),
    NMP(
      name='extra_tree',
      model=ExtraTreeClassifier(random_state=42),
      params=dict(criterion=['gini', 'entropy'],    
                  max_depth=[2, 3, 5, 8, 10],
                  max_features=[0.7, 'sqrt', 'log2'],
                  ccp_alpha=[.0, .3, .7],)
    ),
    NMP(
      name='nys_svm',
      model=Pipeline([("nys", Nystroem(random_state=42)),
            ("svm", LinearSVC(dual=False))]),
      params=dict(svm__C=[1,10,100],    
                  nys__kernel=['poly', 'rbf', 'sigmoid'],)
    ),

    # NMP(
    #   name='svm',
    #   model=SVC(random_state=42, cache_size=400),
    #   params=dict(C=[1,10,100],    
    #               kernel=['poly', 'rbf', 'sigmoid'],
    #               class_weight=[None, 'balanced'],
    #               )
    # ),

    NMP(
      name='rf',
      model=RandomForestClassifier(random_state=42),
      params=dict(max_depth=[3, 5, 8, None],
                  n_estimators=[10, 100, 200, 300],
                  min_samples_split=[2, 5, 9, 11],
                  criterion=["gini", "entropy"],
                  bootstrap=[True, False],),
    ),
    NMP(
      name='gaussian_nb',
      model=GaussianNB(),
      params=dict(var_smoothing=[1e-09]),
    ),

    # NMP(
    #   name='sgd_classifier',
    #   model=Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('sgd', SGDClassifier(n_jobs=-1, max_iter=5000)),
    #   ]),
    #   params=dict(sgd__loss=['hinge', 'modified_huber'],
    #               sgd__learning_rate=['optimal', 'adaptive'],
    #               sgd__eta0=[.001]),
    # ),

    # NMP(
    #   name='knn',
    #   model=KNeighborsClassifier(),
    #   params=dict(
    #     n_neighbors=[3, 5, 8, 10],
    #     weights=['uniform', 'distance'],        
    #   )
    # ),
    # NMP(
    #   name='complement_nb',
    #   model=ComplementNB(),
    #   params=dict(alpha=[.1, .5, 1.,],
    #               norm=[True, False])
    # ),    
    # NMP(
    #   name='logistic_regression',
    #   model=LogisticRegression(random_state=42, n_jobs=-1),
    #   params=dict(
    #           max_iter=[100, 150, 200],
    #           multi_class=['ovr', 'multinomial'],
    #         )
    # ),    
  ]
  if not args.fine_tune:
    print("leaving default parameters...")
    for i in range(len(m_p)):
      m_p[i].params.clear()

  lgb_params = {
    'boosting_type': 'dart',
    'learning_rate': 0.01,
    'tree_learner': 'feature',
    'metric': 'multi_error,multi_logloss',
    'objective': 'multiclassova',
    'num_class': len(m_p),
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
  print("RUN:", run.info.run_id)
  ### LOAD DATA AND FINE TUNE TO INITIAL DATA
  print("[FINETUNING BASE MODELS]")
  df = pd.read_csv(path +'data.csv')
  path += ('unsup_' if args.unsup_features else 'sup_')
  # df = df.iloc[:800]

  from sklearn.preprocessing import LabelEncoder as LE
  df[args.target] = LE().fit_transform(df[args.target])
  df.drop(['date'], axis=1, inplace=True, errors='ignore')

  if not args.temporal_feats:
    df_temporal = pd.DataFrame()
  else:
    import tsfel
    cfg_file = tsfel.get_features_by_domain()
    df_temporal = tsfel.time_series_features_extractor(
      cfg_file, df, window_size=300, 
      overlap=299/300, fs=1,
    )

  if args.fine_tune:
    fine_tune(df, df_temporal, args.initial, args.gamma, 
          args.omega, m_p, args.target, args.eval_metric)
    for tup in m_p:
      mlflow.sklearn.log_model(
        tup.model, '{}__{}.pkl'.format(tup.name, run.info.run_id)
      )

  # dump(m_p, path + 'm_p.joblib')
  # m_p = load(path + 'm_p.joblib')


  ### GENERATE METAFEATURES AND BEST CLASSIFIER FOR INITIAL DATA
  print("[GENERATE METAFEATURE]")
  metadf = []
  if args.quick:
    print("gotta go fassst")
    sup_mfe = MFE(
      groups=['general'], features=['sd', 'min', 'max'], 
      num_cv_folds=1, summary=('mean',))
    unsup_mfe = None
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
    unsup_mfe = MFE(groups=["statistical"], random_state=42)
  off_scores = []

  # This loop generates the data composing the
  # initial data, which means the initial metabase

  # Se nao eh modo rapido OU [eh modo rapido e] nao tem metabase criada, crie-a
  if not args.cache:
  # if (not args.fine_tune or not os.path.isfile(path + 'metabase_fine_tune.csv')) and not args.cache:
    scores_dict_lis = []
    for idx in tqdm(range(0, args.initial)):
      mfe_feats, scores, score_dict = base_train_test(m_p, df, idx, sup_mfe, args.gamma,
                      args.omega, args.target,
                      args.eval_metric, unsup_mfe=unsup_mfe)

      # mfe_feats = [[mfe_feats[k] for k in mfe_feats if k\
      #       not in missing_columns]]
      # print("score_dict:", score_dict)
      # input()
      scores_dict_lis.append(score_dict)

      off_scores.append(scores)

      mfe_feats[metay_label] = np.argmax(scores)
      metadf.append(mfe_feats)
    
    df_scores = pd.DataFrame(scores_dict_lis)
    df_scores.to_csv(path + 'scores_generate_metafeatures.csv')
  
    dump(off_scores, path + 'off_scores.joblib')
    base_data = pd.DataFrame(metadf)
    
    if args.fine_tune:
      print("DUMPED fine_tune_METABASE")
      base_data.to_csv(path + 'metabase_fine_tune.csv', index=False)
    else:
      print("DUMPED FULL_DATASET")
      base_data.to_csv(path + 'metabase.csv', index=False)
  else:    
    print("READING CACHED METABASE...")
    # input()
    if args.fine_tune:
      base_data = pd.read_csv(path + 'metabase_fine_tune.csv')
    else:
      base_data = pd.read_csv(path + 'metabase.csv')

  # print("BASE_DATA.columns:", base_data.columns)
  # input()
  # print("Frequency statistics in metabase:")
  for idx, count in base_data[metay_label].value_counts().items():
    print("\t{:25}{:.3f}".format(str(m_p[idx].name).split('(')[0],
                   count/args.initial))

  ### DROP MISSING DATA AND TRAIN METAMODEL
  print("[OFFLINE LEARNING]")
  missing_columns = base_data.columns[base_data.isnull().any()].values
  print('qtd columns:', len(base_data.columns))
  print('missing columns:', len(missing_columns))
  print('missing columns:', missing_columns)
  
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
  if args.save_metamodel:
    metas.save_model(path + 'metamodel.txt')
  

  # metas = lgb.Booster(model_file=path + 'metamodel.txt')
  print("Kappa:  {:.3f}+-{:.3f}".format(np.mean(kappas), np.std(kappas)))
  print("GMean:  {:.3f}+-{:.3f}".format(np.mean(gmeans), np.std(gmeans)))
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
  print(f"DEFAULT: {default} ({m_p[default].name})", default)
  # input()
  print("BASE_DATASHAPE:", base_data.shape)
  metadf = np.empty((0, base_data.shape[1]-1), np.float32)
  metay = []
  counter = 0

  m_recommended = []
  m_best = []
  m_diff = []
  f_importance = []

  score_recommended = []
  score_default = []
  
  scores_lis = []
  
  small_data = 5000000
  until_data = min(args.initial + small_data,
           int((df.shape[0]-args.omega)/args.gamma))
  for idx in tqdm(range(args.initial, until_data if not args.quick else (args.initial + 50))):
    mfe_feats, scores, score_dict = base_train_test(m_p, df, idx, sup_mfe, args.gamma,
                     args.omega, args.target,
                     args.eval_metric, unsup_mfe=unsup_mfe)
    scores_lis.append(score_dict)

    score_dict['default'] = score_dict[m_p[default].name]
    score_dict['default_model'] = m_p[default].name

    mfe_feats = [[mfe_feats[k] for k in mfe_feats if k\
           not in missing_columns]]
    
    metadf = np.append(metadf, mfe_feats, axis=0)
    max_score = np.argmax(scores)
    meta_pred = metas.predict(mfe_feats)
    recommended = np.argmax(meta_pred[0])

    score_dict['recommended'] = score_dict[m_p[recommended].name]
    score_dict['recommended_model'] = m_p[recommended].name

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
  
  scores_path = path + f'scores_{"|".join([i.name for i in m_p])}.csv'
  
  scores_df = pd.DataFrame(scores_lis)
  scores_df.to_csv(scores_path, index=False)
  mlflow.log_artifact(scores_path)


  dump(m_recommended, path + 'recommended.joblib')
  dump(m_best, path + 'best.joblib')
  dump(m_diff, path + 'difference.joblib')
  dump(score_recommended, path + 'score_reco.joblib')
  dump(score_default, path + 'score_def.joblib')

  # TODO: dump do dataframe com os scores dos classificadores


  dump(metadf, path + 'metadf.joblib')
  dump(f_importance, path + 'tfi.joblib')
  print("Kappa: ", cohen_kappa_score(m_best, m_recommended))
  print("GMean: ", geometric_mean_score(m_best, m_recommended))
  print("Accuracy: ", accuracy_score(m_best, m_recommended))
  print(classification_report(m_best, m_recommended))
  print(classification_report_imbalanced(m_best, m_recommended))
