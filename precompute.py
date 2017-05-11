import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from collections import Counter
import operator

def reduce_class(x,y,y_test=[],threshold = 0.01,other=False):
  print("Reduce Class")
  y_size = len(y)
  freq_y = Counter(y)
  freq_y = sorted(freq_y.items(), key=operator.itemgetter(1),reverse=True)
  SUM = 0
  count = 0
  removed_class = []
  ## fy[0] = class, fy[1] = freq
  for fy in freq_y:
    freq = fy[1]/y_size
    if(freq < threshold):
      count += 1
      SUM += fy[1]
      removed_class.append(fy[0])
      # print(fy,freq)

  print("exist class : ",len(freq_y)-count)
  print("remove amount : ",SUM)
  print("remove rate : ",SUM/y_size)
  print("removed class\n",removed_class)
  for i in range(y_size):
    if y[i] in removed_class:
      if(other):     ############ other
        y[i] = 9999.0
      else:
        y[i] = None
        x[i] = None
  if(other):         ############ other
    for i in range(len(y_test)):
      if y_test[i] in removed_class:
        y_test[i] = 9999.0
  x = x[~np.isnan(x).all(1)]
  y = y[~np.isnan(y)]
  if(other):
    return x,y,y_test
  else:
    return x,y

def feature_selection(train,test,threshold = 0.9):
  print("PCA")
  pca = PCA(svd_solver='full',random_state=2000)
  pca = pca.fit(train)
  explained_variance = pca.explained_variance_ratio_
  SUM = 0
  n_components = 0
  for var in explained_variance:
    if(SUM <= threshold):
      SUM += var
      n_components += 1
    else:  
      break
  print(SUM,n_components)
  pca = PCA(n_components = n_components, svd_solver='full', random_state=2000)
  train = pca.fit_transform(train)
  test = pca.transform(test)
  return train,test
# def feature_selection(train,test):
#   score_before = 0
#   rc_before = 0
#   best_rc = 1
#   best_n_component = 0
#   for i in range(10,100,5):
#     pca = PCA(n_components=int(np.shape(train)[1]*(i/100.0)),svd_solver='full',random_state=2000)
#     score = np.mean(cross_val_score(pca, train))
#     rc = (score-score_before)/score
#     rrc = (rc-rc_before)/rc
#     print((i/100.0),"\t",score,"\t",rc,"\t",rrc)

#     if rc < 0.01:
#       break
#     else:
#       best_rc,best_n_component = rc,(i/100.0)
#     rc_before = rc
#     score_before = score
#   print(best_n_component,"\t",best_rc)
#   pca = PCA(n_components=int(np.shape(train)[1]*best_n_component),svd_solver='full',random_state=2000)
#   train = pca.fit_transform(train)
#   test = pca.transform(test)
#   return train,test
