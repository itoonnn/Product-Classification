import numpy as np
import pandas as pd
from process import *
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from precompute import *

# giant,redmart sift nb**,lr,nn*,svm-lr*,svm-rbf*,svm-poly*
# giant,coldstorage orb nb,lr,nn,svm-lr,svm-rbf,svm-poly
# half(n) "coldstorage","fairprice","giant","redmart" surf nb,lr,nn,svm-lr,svm-rbf,svm-poly
TEST_SIZE = 0.2
SEED = 2000
STORE = ["coldstorage","fairprice","giant","redmart"]
FUNCTION = ["contextual","sift","surf","orb"]
num_k = ["contextual","sqrt(n)","sqrt(half(n))"]
CLASSIFIER = ['Naivebayes','LogisticRegression','NeuralNetwork','SVM-Linear','SVM-RBF','SVM-Poly']
PATH = "image_feature/"

# Specify input csv file
print("STORE")
print("coldstorage == 0")
print("fairprice == 1")
print("giant == 2")
print("redmart == 3")
STORE = STORE[int(input())]

print("FUNCTION")
print("contextual == 0")
print("sift == 1")
print("surf == 2")
print("orb == 3")
FUNCTION = FUNCTION[int(input())]




if(FUNCTION in ["sift","surf","orb"]):
  print("Root n : 1 or Root half n : 2")
  num_k = num_k[int(input())]
else:
  num_k = num_k[0]

PATH += num_k


print("GROUP")
GROUP = int(input())

# for STORE in ['giant','redmart']:
for GROUP in range(0,10): 
  file_train = PATH+"/"+"_".join(["feature",STORE,FUNCTION,"train",str(GROUP)])+".csv"
  file_test = PATH+"/"+"_".join(["feature",STORE,FUNCTION,"test",str(GROUP)])+".csv"
  file_train_label = PATH+"/"+"_".join(["label",STORE,FUNCTION,"train",str(GROUP)])+".csv"
  file_test_label = PATH+"/"+"_".join(["label",STORE,FUNCTION,"test",str(GROUP)])+".csv"
  train = np.loadtxt(file_train,delimiter=',')
  test = np.loadtxt(file_test,delimiter=',')
  label_train = np.loadtxt(file_train_label,delimiter=',')
  label_test = np.loadtxt(file_test_label,delimiter=',')
  print("+++++++++++++ seed",GROUP,"++++++++++" )
  print("Shape of Data")
  print(np.shape(train))
  # print("Classification Test Before")
  # clf = SVC(C=1,probability=True,random_state=SEED,kernel='linear').fit(train, label_train)
  # pred = clf.predict(train)
  # probas_ = clf.predict_proba(train)
  # acc,precision,recall,fbeta,auc_score = getResult(pred,label_train,probas_)
  # print("TRAIN RESULT")
  # print("accuracy :",acc)
  # print("AUROC :",auc_score)
  # acc_train_before,auc_train_before = acc,auc_score

  # pred = clf.predict(test)
  # probas_ = clf.predict_proba(test)
  # acc,precision,recall,fbeta,auc_score = getResult(pred,label_test,probas_)
  # print("TEST RESULT")
  # print("accuracy :",acc)
  # print("AUROC :",auc_score)
  # acc_test_before,auc_test_before = acc,auc_score

  ############### pre-processed data ###################

  train,label_train = reduce_class(train,label_train,threshold=0.001)
  train,test = feature_selection(train,test,threshold=1)
  # feature_selection(train,test)
  ######################################################
  print("Pre-processed Data")
  print(np.shape(train))
  print("Classification Test After")
  clf = SVC(C=1,probability=True,random_state=SEED,kernel='linear').fit(train, label_train)
  pred = clf.predict(train)
  probas_ = clf.predict_proba(train)
  acc,precision,recall,fbeta,auc_score = getResult(pred,label_train,probas_)
  print("TRAIN RESULT")
  print("accuracy :",acc)
  print("AUROC :",auc_score)
  acc_train_after,auc_train_after = acc,auc_score

  pred = clf.predict(test)
  probas_ = clf.predict_proba(test)
  acc,precision,recall,fbeta,auc_score = getResult(pred,label_test,probas_)
  print("TEST RESULT")
  print("accuracy :",acc)
  print("AUROC :",auc_score)
  acc_test_after,auc_test_after = acc,auc_score
  columns = [
    'seed',
    'acc_train_after',
    'auc_train_after',
    'acc_test_after',
    'auc_test_after'
  ]
  result = pd.DataFrame(pd.Series({
    'seed':GROUP,
    'acc_train_after':acc_train_after,
    'auc_train_after':auc_train_after,
    'acc_test_after':acc_test_after,
    'auc_test_after':auc_test_after
  }))
  fname = "RESULT_"+STORE+"_PCA_0-1_class_0-01.csv"
  if(not os.path.isfile(fname)):
    result.to_csv(fname)
    print("create ",fname)
  else:
    exist = pd.DataFrame.from_csv(fname)
    with open(fname, 'a') as f:
      result.to_csv(f, header=False)
