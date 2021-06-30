# TODO: RENOMEAR scorer --> metric, para não confundir as coisas

from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import List, Tuple, Callable

from collections import namedtuple                                      

# import sklearn.metrics as M
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from sklearn.metrics import cohen_kappa_score, mean_squared_error,\
    classification_report, accuracy_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


NMP = namedtuple('NameModelParam', 'name model params')


def split(lis, idx):
  return lis[:idx], lis[idx:]


# def train_sel_split(df, left, mid, right, target):
#   train, test = df.iloc[left:mid], df.iloc[mid:right]
#   xtrain, ytrain = train.drop(target, axis=1), train[target]
#   xtest, ytest = test.drop(target, axis=1), test[target]

#   return xtrain, xtest, ytrain, ytest



def cohen_kappa_score_(targets, preds):
  if np.array_equal(targets, preds):
    return 1.0
  else:
    return cohen_kappa_score(targets, preds)


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
    'cohen_kappa': cohen_kappa_score_,
    'geometric_mean': geometric_mean_score,
    'accuracy': accuracy_score,
  }
  
  @staticmethod
  def _train_sel_split(x, y , l, m, r):
    return x.iloc[l:m], x.iloc[m:r], y.iloc[l:m], y.iloc[m:r]


  def __init__(self, meta_estimator, models: List[NMP], 
    meta_extractor: Callable = meta_extractor, search='random'):
    """[summary]

    Args:
        meta ([type]): [description]
        models (List[NMP]): [description]
        meta_extractor (Callable): [an procedure to generate metafeatures given data]
    """
    self.meta_estimator = meta_estimator
    self.int_to_model = {idx: m.name for (idx, m) in enumerate(models)}
    self.model_to_int = {m.name: idx for (idx, m) in enumerate(models)}
    self.models = models

    self.meta_extractor = meta_extractor
    self.metabase = None
    self.metabase_scores = None
    self.search = search.lower()

  def _fine_tune(self, X, Y, initial, gamma, omega, scorer):
    datasize = omega * initial // gamma - 1
    # fine_tune
    x, y = X.loc[:datasize], Y.loc[:datasize]

    for tup in tqdm(self.models, total=len(self.models)):
      if self.search == 'grid':          
        optmizer = GridSearchCV(
          tup.model, tup.params, scoring=scorer,
          cv=gamma, n_jobs=-1
        )
      elif self.search == 'random':
        optmizer = RandomizedSearchCV(tup.model, tup.params, random_state=42,
                    scoring=scorer, cv=gamma, n_jobs=-1)

      optmizer.fit(x, y)
      tup.model.set_params(**optmizer.best_params_)


  def _build_metabase(self, X, Y, gamma, omega, metric):
    meta_ft_lis = []
    scores_dict_lis = []
    print("Building metabase..")
    rev_map = dict(
      [(t.name, idx) for (idx, t) in enumerate(self.models)]
    )
    for idx in tqdm(range(100)):
      # print(idx)
      left = idx * gamma
      mid = left + omega
      right = mid + gamma

      xtrain, xsel, ytrain, ysel = self._train_sel_split(
        X, Y, left, mid, right
      )

      mfe_feats = self.meta_extractor(xtrain, ytrain)
      # extracao de metafeatures quase pronta, mas falta
      # saber qual o melhor modelo.
      # print(mfe_feats)
      score_dict = {}
      #TODO: adicionar rótulo de melhor algoritmo (target)
      ma, ma_name = -1, None
      for tup in self.models:
        tup.model.fit(xtrain, ytrain)
        pred = tup.model.predict(xsel)
        score_dict[tup.name] = metric(ysel, pred)
        
        if score_dict[tup.name] > ma:
          ma = score_dict[tup.name]
          ma_name = rev_map[tup.name]
      mfe_feats['meta_label'] = ma
      mfe_feats['meta_label_model'] = int(ma_name)

      scores_dict_lis.append( score_dict )
      lis = list(score_dict.items())

      meta_ft_lis.append(mfe_feats)
    
    meta_df = pd.DataFrame(meta_ft_lis)
    scores_df = pd.DataFrame(scores_dict_lis)
    return meta_df, scores_df


  def _offline_learning(self):
    # TODO: relatar métricas especificadas em OFFLINE_LEARNING.
    return


  def fit(self, X, Y, initial, gamma, omega, metric, fine_tune=True):
    # This implements the offline stage of metastream.
    # Steps are as follows:
    # 1- hyper parameter tuning
    # 2 - generate metabase

    self._fine_tune(X, Y, initial, gamma, omega, make_scorer(metric))
    
    self.metabase, self.metabase_scores = self._build_metabase(
      X, Y, gamma=gamma,omega=omega, metric=metric, 
    )
    # self._offline_learning()
    # Gen_metabase

    return self


  def predict(self, X):
    return self.predict(X)
  
  def online_predict():
    pass


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

def online(metamodel, models, data: pd.DataFrame , is_experiment=True):
  # for 
  metamodel.predict()
  if is_experiment:
    for m in models:
      m.fit(X, Y)
    



def metastream():
  offline()
  online()



