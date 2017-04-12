import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score ,recall_score, roc_curve, auc, roc_auc_score,make_scorer,precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import time

def GRID_FN(power):
  return 10**(power)

def roc_auc_fixed(y,y_pred):
  pos_label = max(y)
  fpr, tpr, thresholds = roc_curve(y,y_pred,pos_label=pos_label)
  auc_score = auc(fpr,tpr)
  return auc_score
roc_auc_my = make_scorer(roc_auc_fixed)

def getResult(predictions,labelData,probas_):
  pos_label = max(labelData)
  acc = accuracy_score(labelData,predictions)
  (precision,recall,fbeta,support) = precision_recall_fscore_support(labelData,predictions,pos_label=pos_label,average="weighted")
  # report = classification_report(labelData,predictions)
  auc_score = roc_auc_fixed(labelData,predictions)
  return acc,precision,recall,fbeta,auc_score

def predict_process(data,labelData,clf):
  t = time.time()
  predictions = clf.predict(data)
  predict_time = time.time() - t

  probas_ = clf.predict_proba(data)
  acc,precision,recall,fbeta,auc_score = getResult(predictions,labelData,probas_)
  return acc,precision,recall,fbeta,auc_score,predict_time

def str2float(i):
  if type(i)==str and len(i)>3:
    return tuple(float(j) for j in i[1:-1].split(','))
  else:
    return float(i)
def str2int(i):
  if type(i)==str and len(i)>3:
    return tuple(int(j) for j in i[1:-1].split(','))
  else:
    return int(i)
def list_str2num(list):
  return [ str2float(i) if len(i)>3 else float(i) for i in list]

def classification(name,param,SEED=2000):
  print(name)
  if(name == "NeuralNetwork"):
    if(type(param)!=tuple):
      param = str2int(param)
    return MLPClassifier(hidden_layer_sizes=param,random_state=SEED)
  elif(name == "LogisticRegression"):
    return LogisticRegression(random_state=SEED,C=param,n_jobs=-1,multi_class='multinomial',solver='sag')
  elif(name == "Naivebayes"):
    return MultinomialNB(alpha=param)
  elif(name == "SVM-Linear"):
    return SVC(C=param,kernel='linear',probability=True,random_state=SEED,cache_size=30000)
  elif(name == "SVM-RBF"):
    if(type(param)!=tuple):
      param = str2float(param)
    return SVC(C=param[1],gamma=param[0],kernel='rbf',probability=True,random_state=SEED,cache_size=30000)
  elif(name == "SVM-Poly"):
    if(type(param)!=tuple):
      param = str2float(param)
    return SVC(C=param[1],degree=param[0],kernel='poly',probability=True,random_state=SEED,cache_size=30000)

def init_params(classifier):
  print(classifier)
  if(classifier == "NeuralNetwork"):
    return [(i) for i in range(10,110,10)]+[(i,i) for i in range(10,110,10)]+[(i,i,i) for i in range(10,110,10)]
  elif("SVM-RBF" == classifier):
    return [(GRID_FN(j),GRID_FN(i)) for j in range(-5,6) for i in range(-5,6)]
  elif("SVM-Poly" == classifier):
    return [(j,GRID_FN(i)) for j in range(1,6) for i in range(-5,6)]
  else:
    return [GRID_FN(i) for i in range(-5,6)]
def params_interpreter(data,classifier='default'):
  if("NeuralNetwork" == classifier):
    parameter_exist = list_str2num(data)
    return [ tuple(int(j) for j in i ) if type(i)==tuple else int(i) for i in parameter_exist]
  elif(classifier in ["SVM-RBF","SVM-Poly"]):
    return list_str2num(data)
  else:
    return list(data)
def param_interpreter(rec,classifier='default'):
  if("NeuralNetwork" == classifier):
    return str2int(rec)
  elif(classifier in ["SVM-RBF","SVM-Poly"]):
    return str2float(rec)
  else:
    return rec

def image_classification_process(train,test,label_train,label_test,SEED=2000,GROUP=0,classifier="Naivebayes",feature='sift',numk='sqrt(n)'):
# SEED = 2000
# GROUP = 0-9
# classifier = ['NeuralNetwork','LogiticRergression','Naivebayes','SVM-Linear','SVM-RBF','SVM-Poly']
# feature = ['contextual','sift','surf','orb']
# numk = ['contexual','sqrt(n)','sqrt(half(n))']
  print("CLASSIFICATION with "+feature+" "+numk+" by using "+classifier)
  filename = "_".join(["crossval",classifier,feature,numk])+".csv"
  parameter_all = init_params(classifier)

  print("All param")
  print(parameter_all)

  columns = ['seed','param','score','std']
  df = pd.DataFrame(columns = columns)
  parameter_exist = []
  if(not os.path.isfile(filename)):
    df.to_csv(filename)
    print("create ",filename)
  else:
    data = df.from_csv(filename)
    print("Exist param")

    parameter_exist = params_interpreter(data['param'],classifier=classifier)
    print(parameter_exist)

    print("Run param")
  parameter_run = list(set(parameter_all) - set(parameter_exist))
  print(parameter_run)
  #tune parameter
  for param in parameter_run:
    #cross validation
    print("PARAM ",param)
    clf = classification(classifier,param,SEED=SEED)
    scores = cross_validation.cross_val_score(clf,train,label_train,cv=10,scoring=roc_auc_my)
    score = round(scores.mean(),3)
    std = round(scores.std(),3)
    print(GROUP,param,score,std)
    df = pd.DataFrame([{'seed':GROUP,'param':param,'score':score,'std':std}],columns = columns)
    ### write file
    with open(filename, 'a') as f:
      df.to_csv(f, header=False)
      print("write\n",df)
  data = df.from_csv(filename)
  parameter_exist = params_interpreter(data['param'],classifier=classifier)
  if list(set(parameter_all) - set(parameter_exist)) == []:
    print("\nComplete all parameter =*=*=*=\n")
    rec_max = data.sort(['score','std'],ascending=[0,1]).head(1)
    param_max = param_interpreter(rec_max['param'][0],classifier=classifier)
    ### prediction start ###
    ## classified training
    clf = classification(classifier,param_max,SEED=SEED)
    t = time.time()
    clf.fit(train, label_train)
    training_time = time.time()
    # Check accuracy but this is based on the same data we used for training
    # Use classifier to train and test
    print('\nResult with '+classifier+' ===\n')
    train_acc,train_precision,train_recall,train_f1,train_auc,predict_train_time = predict_process(train,label_train,clf)
    test_acc,test_precision,test_recall,test_f1,test_auc,predict_test_time = predict_process(test,label_test,clf)
    columns = [
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
      'predict_train_time',
      'predict_test_time',
      'training_time',
    ]
    result = pd.DataFrame({
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
      'predict_train_time':predict_train_time,
      'predict_test_time':predict_test_time,
      'training_time':training_time,
    },columns=columns,index=[0])
    print(result)
    return result
  return 0