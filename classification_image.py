import numpy as np
import pandas as pd
from extractImage import *
from process import *

TEST_SIZE = 0.2
SEED = 2000
STORE = ["coldstorage","fairprice","giant","redmart"]
FUNCTION = ["contextual","sift","surf","orb"]
num_k = ["contextual","sqrt(n)","sqrt(half(n))"]
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

print("CLASSIFIER")
print("Naive Bayes == 0")
print("Logitic Regression == 1")
print("Neural Network == 2")
print("SVM-Linear == 3")
print("SVM-RBF == 4")
print("SVM-Poly == 5")
CLASSIFIER = int(input())

for i in range(10):
  file_train = PATH+"/"+"_".join(["feature",STORE,FUNCTION,"train",str(i)])+".csv"
  file_test = PATH+"/"+"_".join(["feature",STORE,FUNCTION,"test",str(i)])+".csv"
  file_train_label = PATH+"/"+"_".join(["label",STORE,FUNCTION,"train",str(i)])+".csv"
  file_test_label = PATH+"/"+"_".join(["label",STORE,FUNCTION,"test",str(i)])+".csv"
  train = np.loadtxt(file_train,delimiter=',')
  test = np.loadtxt(file_test,delimiter=',')
  label_train = np.loadtxt(file_train_label,delimiter=',')
  label_test = np.loadtxt(file_test_label,delimiter=',')
  if(CLASSIFIER == 0):
    naivebeys_process(SEED,i,-5,5,train,test,label_train,label_test)


##### classification
# clf = SVC(C=1,probability=True,random_state=SEED).fit(train, label_train)

# pred = clf.predict(train)
# probas_ = clf.predict_proba(train)
# acc,precision,recall,fbeta,auc_score = getResult(pred,label_train,probas_)
# print("TRAIN RESULT")
# print("accuracy :",acc)
# print("precision :",precision)
# print("recall :",recall)
# print("f-score :",fbeta)
# print("AUROC :",auc_score)