import numpy as np
import pandas as pd
from extractImage import *
from process import *
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

print("CLASSIFIER")
print("SVM-Linear C=1")
CLASSIFIER = CLASSIFIER[3]

print("GROUP")
GROUP = int(input())

# for GROUP in range(0,9): 
file_train = PATH+"/"+"_".join(["feature",STORE,FUNCTION,"train",str(GROUP)])+".csv"
file_test = PATH+"/"+"_".join(["feature",STORE,FUNCTION,"test",str(GROUP)])+".csv"
file_train_label = PATH+"/"+"_".join(["label",STORE,FUNCTION,"train",str(GROUP)])+".csv"
file_test_label = PATH+"/"+"_".join(["label",STORE,FUNCTION,"test",str(GROUP)])+".csv"
train = np.loadtxt(file_train,delimiter=',')
test = np.loadtxt(file_test,delimiter=',')
label_train = np.loadtxt(file_train_label,delimiter=',')
label_test = np.loadtxt(file_test_label,delimiter=',')
print("Shape of Data")
print(np.shape(train))
print(np.shape(test))
############### pre-processed data ###################
train,label_train,label_test = reduce_class(train,label_train,threshold=0.01,other=True,y_test=label_test)

######################################################
print("Pre-processed Data")
print(np.shape(train))
print(np.shape(test))
classifier = CLASSIFIER
clf = classification(classifier,1,SEED=SEED)
clf.fit(train, label_train)
train_acc,train_precision,train_recall,train_f1,train_auc,predict_train_time = predict_process(train,label_train,clf)
test_acc,test_precision,test_recall,test_f1,test_auc,predict_test_time = predict_process(test,label_test,clf)
columns = [
  'store',
  'seed',
  'classifier',
  'feature',
  'numk',
  'param',
  'train_acc',
  'train_precision',
  'train_recall',
  'train_f1',
  'train_auc',
  'test_acc',
  'test_precision',
  'test_recall',
  'test_f1',
  'test_auc',
]
result = pd.DataFrame({
  'store':store,
  'seed':GROUP,
  'classifier': classifier,
  'feature': feature,
  'numk': numk,
  'param':str(param_max),
  'train_acc':train_acc,
  'train_precision':train_precision,
  'train_recall':train_recall,
  'train_f1':train_f1,
  'train_auc':train_auc,
  'test_acc':test_acc,
  'test_precision':test_precision,
  'test_recall':test_recall,
  'test_f1':test_f1,
  'test_auc':test_auc,
},columns=columns,index=[0])

print(result)
fname = "RESULT_IMAGE_CLASSIFICATION.csv"
if(not os.path.isfile(fname)):
  result.to_csv(fname)
  print("create ",fname)
else:
  exist = pd.DataFrame.from_csv(fname)
  if(len(exist.loc[(exist['store'] == STORE) & (exist['seed'] == GROUP) & (exist['classifier'] == CLASSIFIER) & (exist['feature'] == FUNCTION) & (exist['numk'] == num_k) ])==0):
    with open(fname, 'a') as f:
      result.to_csv(f, header=False)
      print("Saved")
  else:
    print("Already Exist")

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