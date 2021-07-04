# TODO: RENOMEAR scorer --> metric, para não confundir as coisas

import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import List, Tuple, Callable

from collections import namedtuple                                      

# import sklearn.metrics as M
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import cohen_kappa_score, mean_squared_error,\
    classification_report, accuracy_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import metastream.helper as H
import lightgbm as lgb
import mlflow

# NMP = namedtuple('NameModelParam', 'name model params')


lgb_params = {
  'boosting_type': 'dart',
  'learning_rate': 0.01,
  'tree_learner': 'feature',
  'metric': 'multi_error,multi_logloss',
  'objective': 'multiclassova',
  'num_class': 2,
  # 'metric': 'binary_error,binary_logloss',
  # 'objective': 'binary',
  'is_unbalance': True,
  'verbose': -1,
  'seed': 42
}

# def train_sel_split(df, left, mid, right, target):
#   train, test = df.iloc[left:mid], df.iloc[mid:right]
#   xtrain, ytrain = train.drop(target, axis=1), train[target]
#   xtest, ytest = test.drop(target, axis=1), test[target]

#   return xtrain, xtest, ytrain, ytest



def meta_extractor(xdata, ydata, sup_mfe, unsup_mfe=None):
  mfe_feats = {}
  sup_mfe.fit(xdata.values, ydata.values)
  if unsup_mfe:
    unsup_mfe.fit(xtrain.values)
    unsup_feats = {f'unsup_{k}':v for (k,v) in zip(*unsup_mfe.extract())}
    mfe_feats.update(unsup_feats)
  sup_metafeatures = sup_mfe.extract()
  mfe_feats.update( dict(zip(*sup_metafeatures)) )
  return mfe_feats


class MetaStream(BaseEstimator, TransformerMixin):
  OFFLINE_METRICS = {
    'cohen_kappa': H.cohen_kappa_score,
    'geometric_mean': H.geometric_mean_score,
    'accuracy': H.accuracy_score,
  }
  
  @staticmethod
  def _train_sel_split(x, y , l, m, r):
    return x.iloc[l:m], x.iloc[m:r], y.iloc[l:m], y.iloc[m:r]


  def __init__(self, meta_estimator, models: List[dict], 
    meta_extractor: Callable = meta_extractor, search='random'):
    """[summary]

    Args:
        meta ([type]): [description]
        models (List[NMP]): [description]
        meta_extractor (Callable): [an procedure to generate metafeatures given data]
    """
    self.meta_estimator = meta_estimator
    self.int_to_model = {idx: m['name'] for (idx, m) in enumerate(models)}
    self.model_to_int = {m['name']: idx for (idx, m) in enumerate(models)}
    self.models = models

    self.meta_extractor = meta_extractor
    self.metabase = None
    self.metabase_scores = None

    self.offline_metrics = None
    self.offline_y = None

    self.search = search.lower()


  # OK
  def _fine_tune(self, X, Y, initial, gamma, omega, scorer):
    datasize = omega * initial // gamma - 1
    x, y = X.loc[:datasize], Y.loc[:datasize]

    for tup in tqdm(self.models, total=len(self.models)):
      if self.search == 'grid':          
        optmizer = GridSearchCV(
          tup['model'], tup['params'], scoring=scorer,
          cv=gamma, n_jobs=-1
        )
      elif self.search == 'random':
        optmizer = RandomizedSearchCV(tup['model'], tup['params'], random_state=42,
                    scoring=scorer, cv=gamma, n_jobs=-1)

      optmizer.fit(x, y)
      tup['model'].set_params(**optmizer.best_params_)


  # 
  def _build_metabase(self, X, Y, gamma, omega, metric, size, meta_extractor=None):
    meta_extractor = meta_extractor or self.meta_extractor
    meta_ft_lis = []
    scores_dict_lis = []
    print("Building metabase..")

    # for idx in tqdm(range(200)):
    for idx in tqdm(range(size)):
      left = idx * gamma
      right = (idx + 1) * gamma + omega

      x, y = X[left:right], Y[left:right]
      xtrain, xsel = np.split(x, [omega]) 
      ytrain, ysel = np.split(y, [omega]) 

      mfe_feats = self.meta_extractor(xtrain, ytrain)
      score_dict = {}
      ma, ma_name = -np.inf, None

      for tup in self.models:
        tup['model'].fit(xtrain, ytrain)
        pred = tup['model'].predict(xsel)
        score_dict[tup['name']] = metric(ysel, pred)        
        if score_dict[tup['name']] > ma:
          ma = score_dict[tup['name']]
          ma_name = tup['name']

      mfe_feats['meta_best_name'] = ma_name
      mfe_feats['meta_best_classifier'] = ma
      mfe_feats['meta_label'] = int(self.model_to_int[ma_name])

      scores_dict_lis.append( score_dict )
      lis = list(score_dict.items())

      meta_ft_lis.append(mfe_feats)
    
    meta_df = pd.DataFrame(meta_ft_lis)
    scores_df = pd.DataFrame(scores_dict_lis)
    return meta_df, scores_df


  def _offline_learning(self, omega, gamma, test_size, metric_dict, ):
    # TODO: relatar métricas especificadas em OFFLINE_LEARNING.
    # print(sorted(self.metabase.columns))
    print("[OFFLINE LEARNING]")
    missing_columns = self.metabase.columns[self.metabase.isnull().any()].values
    metabase = self.metabase.drop(missing_columns, axis=1)
    mX, mY = metabase.drop(
      columns=metabase.filter(like='meta_', axis=1).columns
    ).values, metabase['meta_label'].values
    off_targets = []
    off_preds = []
    metrics = {k: [] for k in metric_dict}
    for i in tqdm(range(test_size)):
      itest = i+omega
      metas = lgb.train(lgb_params,
                train_set=lgb.Dataset(mX[i:i+omega],
                                      mY[i:i+omega]))
      raw_preds=metas.predict(mX[itest:itest+gamma])
      preds = np.apply_along_axis(np.argmax, 1, raw_preds)   
      targets = mY[itest:itest+gamma]
      for m, fun in metric_dict.items():
        res = fun(y_true=targets, y_pred=preds)
        metrics[m].append( res )
        # mlflow.log_metric(f'off_{m}', res, i)

      off_targets.append(targets)
      off_preds.append(preds)
    print(len(off_targets), len(off_preds))
    return (
      pd.DataFrame(metrics), 
      pd.DataFrame(dict(y_true=off_target, y_preds=off_preds)),
    )


  def fit(self, X, Y, initial, gamma, omega, metric, metabase_size,
    fine_tune=True, offline_eval=False, offline_test_size=100,
    offline_metrics={}, offline_learning_size=100, metabase=None, meta_scores=None):
    # This implements the offline stage of metastream.
    # Steps are as follows:
    # 1 - hyper parameter tuning
    # 2 - generate metabase

    # self._fine_tune(X, Y, initial, gamma, omega, make_scorer(metric))
    
    self.metabase, self.metabase_scores = self._build_metabase(
      X, Y, gamma=gamma,omega=omega, metric=metric, size=metabase_size,
    )
    if offline_eval:
      df_metrics, df_y = self._offline_learning(
        omega=omega, gamma=gamma, test_size=offline_test_size + offline_learning_size,
        metric_dict=offline_metrics,
      )
      self.offline_metrics = df_metrics
      self.offline_y = df_y
    
    # self._offline_learning()
    # Gen_metabase

    return self


  def predict(self, X):
    return self.predict(X)
  
  def online_predict():
    pass

