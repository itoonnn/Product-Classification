import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from collections import Counter
import operator

def reduce_class(x,y,threshold = 0.01):
  label_size = len(y)
  print(label_size)
  flabel = Counter(y)
  flabel = sorted(flabel.items(), key=operator.itemgetter(1),reverse=True)
  SUM = 0
  count = 0
  removed_class = []
  for label in flabel:
    freq = label[1]/label_size
    if(freq < threshold):
      count += 1
      SUM += label[1]
      removed_class.append(label[0])
      print(label,freq)
  print(len(flabel)-count)
  print(SUM)
  print(SUM/label_size)
  print(removed_class)
  for i in range(label_size):
    if y[i] in removed_class:
      y[i] = None
      x[i] = None
  x = x[~np.isnan(x).all(1)]
  y = y[~np.isnan(y)]
  # print(np.shape
  return x,y
def feature_selection(train,test):
  score_before = 0
  rc_before = 0
  best_rc = 1
  best_n_component = 0
  for i in range(10,100,5):
    pca = PCA(n_components=int(np.shape(train)[1]*(i/100.0)),svd_solver='full',random_state=2000)
    score = np.mean(cross_val_score(pca, train))
    rc = (score-score_before)/score
    rrc = (rc-rc_before)/rc
    print((i/100.0),"\t",score,"\t",rc,"\t",rrc)

    if rc < 0.01:
      break
    else:
      best_rc,best_n_component = rc,(i/100.0)
    rc_before = rc
    score_before = score
  print(best_n_component,"\t",best_rc)
  pca = PCA(n_components=int(np.shape(train)[1]*best_n_component),svd_solver='full',random_state=2000)
  train = pca.fit_transform(train)
  test = pca.transform(test)
  return train,test