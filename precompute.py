import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

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