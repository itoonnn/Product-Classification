import numpy as np
import pandas as pd
from process import *
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

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
print("Naive Bayes == 0")
print("Logitic Regression == 1")
print("Neural Network == 2")
print("SVM-Linear == 3")
print("SVM-RBF == 4")
print("SVM-Poly == 5")
CLASSIFIER = CLASSIFIER[int(input())]

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
  print(np.shape(train)[1])
  print("Classification Test Before")
  clf = SVC(C=10,probability=True,random_state=SEED,kernel='linear').fit(train, label_train)
  pred = clf.predict(train)
  probas_ = clf.predict_proba(train)
  acc,precision,recall,fbeta,auc_score = getResult(pred,label_train,probas_)
  print("TRAIN RESULT")
  print("accuracy :",acc)
  print("AUROC :",auc_score)
  pred = clf.predict(test)
  probas_ = clf.predict_proba(test)
  acc,precision,recall,fbeta,auc_score = getResult(pred,label_test,probas_)
  print("TEST RESULT")
  print("accuracy :",acc)
  print("AUROC :",auc_score)
  ############### pre-processed data ###################
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

  ######################################################
  print("Pre-processed Data")
  print(np.shape(train)[1])
  print("Classification Test After")
  clf = SVC(C=10,probability=True,random_state=SEED,kernel='linear').fit(train, label_train)
  pred = clf.predict(train)
  probas_ = clf.predict_proba(train)
  acc,precision,recall,fbeta,auc_score = getResult(pred,label_train,probas_)
  print("TRAIN RESULT")
  print("accuracy :",acc)
  print("AUROC :",auc_score)
  pred = clf.predict(test)
  probas_ = clf.predict_proba(test)
  acc,precision,recall,fbeta,auc_score = getResult(pred,label_test,probas_)
  print("TEST RESULT")
  print("accuracy :",acc)
  print("AUROC :",auc_score)
  # result = image_classification_process(train,test,label_train,label_test,SEED=SEED,GROUP=GROUP,store=STORE,classifier=CLASSIFIER,feature=FUNCTION,numk=num_k)
  # print(result)
  # fname = "RESULT_IMAGE_CLASSIFICATION.csv"
  # if(not os.path.isfile(fname)):
  #   result.to_csv(fname)
  #   print("create ",fname)
  # else:
  #   exist = pd.DataFrame.from_csv(fname)
  #   if(len(exist.loc[(exist['store'] == STORE) & (exist['seed'] == GROUP) & (exist['classifier'] == CLASSIFIER) & (exist['feature'] == FUNCTION) & (exist['numk'] == num_k) ])==0):
  #     with open(fname, 'a') as f:
  #       result.to_csv(f, header=False)
  #       print("Saved")
  #   else:
  #     print("Already Exist")

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