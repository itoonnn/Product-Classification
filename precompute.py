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

  uprint(RenderTree(root,style=AsciiStyle()))

  rootNum = Node('root')
  catNum_top = []
  catNum_sec = {}
  catNum_third = {}

  top_labeler = LabelEncoder()
  top_labels = list([child.name for child in root.children])
  top_numlabels = top_labeler.fit_transform(top_labels)
  top_norm = Normalizer()
  top_numlabels = top_norm.fit_transform(top_numlabels)[0]
  for i,top_node in enumerate(top_level):
    catNum_top.append(Node(top_numlabels[i],parent=rootNum))
    catNum_sec[top_numlabels[i]] = []
    catNum_third[top_numlabels[i]] = {}

    sec_labeler = LabelEncoder()
    sec_labels = list([child.name for child in cat_top[i].children])
    sec_numlabels = sec_labeler.fit_transform(sec_labels)
    sec_norm = Normalizer()
    if(len(sec_numlabels)!=0):
      sec_numlabels=sec_norm.fit_transform(sec_numlabels)[0]

    # uprint(RenderTree(sec_numlabels,style=AsciiStyle()))
    j_num = 0
    for j,sec_node in enumerate(second_level):
      if( len(sec_node)>=2 and sec_node[0]==top_node ):
        catNum_sec[top_numlabels[i]].append(Node(sec_numlabels[j_num],parent=catNum_top[-1]))
        catNum_third[top_numlabels[i]][sec_numlabels[j_num]] = []
        j_num += 1


        third_labeler = LabelEncoder()
        print(cat_top)
        # third_labels = list([child.name for child in cat_top[i].children])
        # third_numlabels = third_labeler.fit_transform(third_labels)
        # third_norm = Normalizer()
        # if(len(sec_numlabels)!=0):
        #   sec_numlabels = sec_norm.fit_transform(sec_numlabels)[0]
        # else:
        #   sec_numlabels = []
  # print(top_level)
  # print(second_level)
  uprint(RenderTree(rootNum,style=AsciiStyle()))


  
  # for pre, fill, node in RenderTree(root):
  #   uprint(u"%s%s" % (pre, node.name))


  # cat_tree = dict()
  # for top in top_level:
  #   cat_tree[top] = dict()
  #   for second in second_level:
  #     if(len(second)>=2 and second[0]==top):
  #       cat_tree[top][second[1]] = list() 
  #       for third in third_level:
  #         if(len(third)>=3 and third[0]==top and third[1]==second[1]):
  #           cat_tree[top][second[1]].append(third[2])

  # top_labeler = LabelEncoder()
  # top_labels = list(cat_tree.keys())
  # top_numlabels = top_labeler.fit_transform(top_labels)
  # cat_tree.keys() = dictkeys(top_numlabels)

  # second_labels = {}
  # second_labeler = {}
  # second_numlabels = {}
  # for top_node in top_labels:
  #   second_labeler[top_node] = LabelEncoder()
  #   second_labels = list(cat_tree[top_node].keys())
  #   second_numlabels[top_node] = second_labeler[top_node].fit_transform(second_labels)

  #   third_labels = {}
  #   third_labeler = {}
  #   third_numlabels = {}
  #   for second_node in second_labels:
  #     third_labeler[second_node] = LabelEncoder()
  #     third_labels = list(cat_tree[top_node][second_node])
  #     third_numlabels[second_node] = third_labeler[second_node].fit_transform(second_labels)

  # print(top_numlabels)
  # print(second_numlabels)
  # print(third_numlabels)


  return 0

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
