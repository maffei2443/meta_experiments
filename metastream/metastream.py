from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import List, Tuple, Callable

from collections import namedtuple                                      

# import sklearn.metrics as M
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from sklearn.metrics import cohen_kappa_score, mean_squared_error
    classification_report, accuracy_score, make_scorer

NMP = namedtuple('NameModelParam', 'name model params')

def cohen_kappa_score_(targets, preds):
  if if np.array_equal(targets, preds):
    return 1.0
  else
    return cohen_kappa_score(targets, preds)




class MetaStream(BaseEstimator, TransformerMixin):
  OFFLINE_METRICS = [
    'cohen_kappa': cohen_kappa_score_,
    'geometric_mean': M.geometric_mean_score,
    'accuracy': M.accuracy_score,
  ]

  def __init__(self, meta, models: List[NMP], meta_extractor: Callable):
    """[summary]

    Args:
        meta ([type]): [description]
        models (List[NMP]): [description]
        meta_extractor (Callable): [an procedure to generate metafeatures given data]
    """
    self.clf = meta
    self.models = models
    self.meta_extractor = meta_extractor
    self.first_metabase = None

  def _fine_tune(self, X, target_col, initial, gamma, omega, scorer):
    datasize = omega * initial // gamma - 1
    # fine_tune
    x, y = X.loc[:datasize].drop([target_col], axis=1), X.loc[:datasize, target]

    for tup in self.models:
      opmizer = GridSearchCV(
        tup.model, tup.params, scoring=scorer,
        cv=gamma, n_jobs=-1
      )
      optimizer.fit(x, y)
      tup.model.set_params(**optimizer.best_params_)


  def _build_metabase(self, X):
    meta_ft_lis = []
    for idx in tqdm(range(0, args.initial)):
        mfe_feats = self.meta_extractor(X)
        meta_ft_lis.append(mfe_feats)

    meta_df = pd.DataFrame(metadf)
    return meta_df

  def _offline_learning(self):
    # TODO: relatar métricas especificadas em OFFLINE_LEARNING.
    return

  def fit(self, X, target_col, initial, gamma, omega, scorer):
    # This implements the offline stage of metastream.
    # Steps are as follows:
    # 1- hyper parameter tuning
    # 2 - generate metabase

    self._fine_tune(X, target_col, initial, gamma, omega, scorer)
    
    self.first_metabase = self._build_metabase(X, target_col)

    self._offline_learning()
    # Gen_metabase

    return self


  def predict(self, X):
    return self.predict(X)
  
  def online_predict():



def fine_tune(data, initial, gamma, omega, models_tup: NMP, 
  target, eval_metric):
  datasize = omega * initial // gamma - 1 # initial base data
  # tmp =
  Xcv, ycv = data.loc[:datasize].drop([target], axis=1),\
    data.loc[:datasize, target]
  for tup in tqdm(models_tup, total=len(models_tup)):
      
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


def offline(models):
  fine_tune()
  build_metabase()

def online(metamodel, models, data: pd.DataFrame, , is_experiment=True):
  for 
  metamodel.predict()
  if is_experiment:
    for m in models:
      m.fit(X, Y)
    



def metastream():
  offline()
  online()



