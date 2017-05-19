#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from anytree import Node, RenderTree, AsciiStyle
from sklearn.preprocessing import Normalizer
import operator,sys

def uprint(*objects, sep=' ', end='\n', file=sys.stdout):
    enc = file.encoding
    if enc == 'UTF-8':
        print(*objects, sep=sep, end=end, file=file)
    else:
        f = lambda obj: str(obj).encode(enc, errors='backslashreplace').decode(enc)
        print(*map(f, objects), sep=sep, end=end, file=file)

def build_heirarchy_label(label):
  cat_map = pd.DataFrame(columns=[
    'level',
    'top_name',
    'second_name',
    'third_name',
    'top_value',
    'second_value',
    'third_value'
  ])
  seq_label = label.str.split("->")
  # [list(x) for x in set(tuple(x) for x in testdata)]
  top_level = set(map(lambda x: tuple(x)[0], seq_label))
  second_level = set(map(lambda x: tuple(x)[0:2], seq_label))
  third_level = set(map(lambda x: tuple(x)[0:3], seq_label))

  root = Node('root')
  cat_top = []
  cat_sec = {}
  cat_third = {}

  for top_node in top_level:
    cat_top.append(Node(top_node,parent=root))
    cat_sec[top_node] = []
    cat_third[top_node] = {}
    for second_node in second_level:
      if( len(second_node)>=2 and second_node[0]==top_node ):
        cat_sec[top_node].append(Node(second_node[1],parent=cat_top[-1]))
        cat_third[top_node][second_node[1]] = []
        for third_node in third_level:
          if( len(third_node)>=3 and third_node[0]==second_node[0] and third_node[1]==second_node[1] ):
            cat_third[top_node][second_node[1]].append(Node(third_node[2],parent=cat_sec[top_node][-1]))


  rootNum = Node('root')
  catNum_top = []
  catNum_sec = {}
  catNum_third = {}

  top_labeler = LabelEncoder()
  top_labels = list([child.name for child in root.children])
  top_numlabels = top_labeler.fit_transform(top_labels)
  top_norm = Normalizer()
  top_numlabels = top_norm.fit_transform(top_numlabels)[0]
  idx = 0
  for i,top_node in enumerate(top_level):
    catNum_top.append(Node(top_numlabels[i],parent=rootNum))
    catNum_sec[top_numlabels[i]] = []
    catNum_third[top_numlabels[i]] = {}
    cat_map.loc[idx] = [0,top_node,None,None,top_numlabels[i],None,None]
    idx+=1
    sec_labeler = LabelEncoder()
    sec_labels = list([child.name for child in cat_top[i].children])
    if(len(sec_labels)!=0):      
      sec_numlabels = sec_labeler.fit_transform(sec_labels)
      sec_norm = Normalizer()
      sec_numlabels=sec_norm.fit_transform(sec_numlabels)[0]

      j_num = 0
      for j,sec_node in enumerate(second_level):
        if( len(sec_node)>=2 and sec_node[0]==top_node ):
          catNum_sec[top_numlabels[i]].append(Node(sec_numlabels[j_num],parent=catNum_top[-1]))
          catNum_third[top_numlabels[i]][sec_numlabels[j_num]] = []
          cat_map.loc[idx] = [1,top_node,sec_node[1],None,top_numlabels[i],sec_numlabels[j_num],None]
          idx+=1
          third_labeler = LabelEncoder()
          third_labels = list([ child.name for child in cat_sec[sec_node[0]][j_num].children])
          if(len(third_labels)!=0):
            third_numlabels = third_labeler.fit_transform(third_labels)
            third_norm = Normalizer()
            third_numlabels = third_norm.fit_transform(third_numlabels)[0]

            k_num = 0
            for k, third_node in enumerate(third_level):
              if(len(third_node)>=3 and third_node[0]==sec_node[0] and third_node[1]==sec_node[1]):
                catNum_third[top_numlabels[i]][sec_numlabels[j_num]].append(Node(third_numlabels[k_num],parent=catNum_sec[top_numlabels[i]][j_num]))
                cat_map.loc[idx] = [2,top_node,sec_node[1],third_node[2],top_numlabels[i],sec_numlabels[j_num],third_numlabels[k_num]]
                idx+=1
                k_num += 1
          j_num += 1
  # uprint(RenderTree(rootNum,style=AsciiStyle()))
  # uprint(RenderTree(root,style=AsciiStyle()))
  return cat_map
def map_label(y,cmap):
  print("mapping label")
  y_top = []
  y_sec = []
  y_third = []
  columns=[
    'level',
    'top_name',
    'second_name',
    'third_name',
    'top_value',
    'second_value',
    'third_value'
  ]
  y_map = pd.DataFrame(columns = columns)
  for i in range(len(y)):
    top_name,second_name,third_name = None,None,None
    top_node,sec_node,third_node = None,None,None
    if(len(y[i])>0):
      top_node = cmap[(cmap['level']==0)&(cmap['top_name']== y[i][0])]['top_value'].values[0] if len(y[i])>0 else None
      level = 0
      top_name = y[i][0]
    if(len(y[i])>1):
      sec_node = cmap[(cmap['level']==1)&(cmap['top_name']== y[i][0])&(cmap['second_name']== y[i][1])][['top_value','second_value']].values[0][1] if len(y[i])>1 else None    
      level = 1
      second_name = y[i][1]
    if(len(y[i])>2):
      third_node =  cmap[(cmap['level']==2)&(cmap['top_name']== y[i][0])&(cmap['second_name']== y[i][1])&(cmap['third_name']== y[i][2])][['top_value','second_value','third_value']].values[0][2] if len(y[i])>2 else None
      level = 2
      third_name = y[i][2]
    # y_third_node = cmap[(cmap['level']==2)&(cmap['top_name']== y[i][0])&(cmap['second_name']== y[i][1])&(cmap['third_name']== y[i][2])][['top_value','second_value','third_value']].values[0] if len(y[i])>2 else None
    # y_sec_node = cmap[(cmap['level']==1)&(cmap['top_name']== y[i][0])&(cmap['second_name']== y[i][1])][['top_value','second_value']].values[0] if len(y[i])>1 else None
    # y_top_node = cmap[(cmap['level']==0)&(cmap['top_name']== y[i][0])]['top_value'].values[0] if len(y[i])>0 else None
    # y_top.append(y_top_node)
    # y_sec.append(y_sec_node)
    # y_third.append(y_third_node)
    

    node = {
    'level':level,
    'top_name':top_name,
    'second_name':second_name,
    'third_name':third_name,
    'top_value':top_node,
    'second_value':sec_node,
    'third_value':third_node
    }
    y_map = y_map.append(node, ignore_index=True)

  # y_top = np.array(y_top)
  # y_sec = np.array(y_sec)
  # y_third = np.array(y_third)
  # print(y_sec[y_sec[:,0]==y_top[0]][:,1]) ## numpy selecting
  # print(y_third[y_third[:,0]==y_top[0]][:,2])
  return y_map


def reduce_class(x,y,threshold = 0.01,other=False):
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

def reduce_hierachical_class(y,threshold = 0.01,other=False):
  print("Reduce Class")
  y = y.reset_index()
  y = y.fillna(99)
  y_size = len(y)
  print(y.groupby(['top_value','second_value','third_value']).size())
  # freq_y = Counter(y)
  # freq_y = sorted(freq_y.items(), key=operator.itemgetter(1),reverse=True)
  return 0
  # SUM = 0
  # count = 0
  # removed_class = []
  # ## fy[0] = class, fy[1] = freq
  # for fy in freq_y:
  #   freq = fy[1]/y_size
  #   if(freq < threshold):
  #     count += 1
  #     SUM += fy[1]
  #     removed_class.append(fy[0])
  #     # print(fy,freq)

  # print("exist class : ",len(freq_y)-count)
  # print("remove amount : ",SUM)
  # print("remove rate : ",SUM/y_size)
  # print("removed class\n",removed_class)
  # for i in range(y_size):
  #   if y[i] in removed_class:
  #     if(other):     ############ other
  #       y[i] = 9999.0
  #     else:
  #       y[i] = None
  #       x[i] = None
  # if(other):         ############ other
  #   for i in range(len(y_test)):
  #     if y_test[i] in removed_class:
  #       y_test[i] = 9999.0
  # x = x[~np.isnan(x).all(1)]
  # y = y[~np.isnan(y)]
  # if(other):
  #   return x,y,y_test
  # else:
  #   return x,y

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
