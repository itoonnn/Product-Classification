import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score ,recall_score, roc_curve, auc, roc_auc_score,make_scorer,precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

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
  predictions = clf.predict(data)
  probas_ = clf.predict_proba(data)
  acc,precision,recall,fbeta,auc_score = getResult(predictions,labelData,probas_)
  return acc,precision,recall,fbeta,auc_score

def str2float(i):
  return tuple(float(j) for j in i[1:-1].split(','))
def str2int(i):
  return tuple(int(j) for j in i[1:-1].split(','))
def list_str2num(list):
  return [ str2float(i) if len(i)>3 else float(i) for i in list]

def classifier(name,param,SEED=2000):
  if(name == "NeuralNetwork"):
    return MLPClassifier(hidden_layer_sizes=param,random_state=SEED)
  elif(name == "LogisticRegression"):
    return LogisticRegression(random_state=SEED,C=param,n_jobs=-1,multi_class='multinomial',solver='sag')
  elif(name == "Naivebayes"):
    return MultinomialNB(alpha=param)
  elif(name == "SVM-Linear"):
    return SVC(C=param,kernel='linear',probability=True,random_state=SEED)
  elif(name == "SVM-RBF"):
    return SVC(C=param[1],gamma=param[0],kernel='rbf',probability=True,random_state=SEED)
  elif(name == "SVM-Poly"):
    return SVC(C=param[1],degree=param[0],kernel='poly',probability=True,random_state=SEED)

def init_param(classifier):
  if("NeuralNetwork" == classifier):
    return [(i) for i in range(10,110,10)]+[(i,i) for i in range(10,110,10)]+[(i,i,i) for i in range(10,110,10)]
  elif("SVM-RBF" == classifier):
    return [(GRID_FN(j),GRID_FN(i)) for j in range(-5,6) for i in range(-5,6)]
  elif("SVM-Poly" == classifier):
    return [(j,GRID_FN(i)) for j in range(1,6) for i in range(-5,6)]
  else:
    return [GRID_FN(i) for i in range(-5,6)]
def param_interpreter(data,classifier='default'):
  if("NeuralNetwork" == classifier)
    parameter_exist = list_str2num(data)
    return [ tuple(int(j) for j in i ) if type(i)==tuple else int(i) for i in parameter_exist]
  elif(classifier in ["SVM-RBF","SVM-Poly"]):
    return list_str2num(data)
  else:
    return data

def image_classification_process(train,test,label_train,label_test,SEED=2000,GROUP=0,classifier="Naivebayes",feature='sift',numk='sqrt(n)'):
# SEED = 2000
# GROUP = 0-9
# classifier = ['NeuralNetwork','LogiticRergression','Naivebayes','SVM-Linear','SVM-RBF','SVM-Poly']
# feature = ['contextual','sift','surf','orb']
# numk = ['contexual','sqrt(n)','sqrt(half(n))']
  print("CLASSIFICATION with "+feature+" "+numk+" by using "+classifier)
  filename = "_".join(["crossval",classifier,feature,numk])+".csv"
  parameter_all = init_param(classifier)

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

    parameter_exist = list_str2num(data['param'])
    print(parameter_exist)

  #tune parameter
  for param in parameter_all:
    if(param not in parameter_exist):
      #cross validation
      clf = classifier(classifier,param,SEED=SEED)
      scores = cross_validation.cross_val_score(clf,train,label_train,cv=10,scoring=roc_auc_my)
      score = round(scores.mean(),3)
      std = round(scores.std(),3)
      print(GROUP,param,score,std)
      df = pd.DataFrame([{'seed':GROUP,'param':param,'score':score,'std':std}],columns = columns)
      ### write file
      with open(filename, 'a') as f:
        df.to_csv(f, header=False)
        print("write\n",df)
  data = df.from_csv(fname)
  parameter_exist = param_interpreter(data['param'])
  if np.array_equal(parameter_all, parameter_exist) or parameter_all == parameter_exist:
    print("Complete all parameter =*=*=*=")
    rec_max = data.sort(['score','std'],ascending=[0,1]).head(1)
    param_max = str2int(rec_max['param'][0])
    ### prediction start ###
    ## classified training
    clf = classifier(classifier,param_max,SEED=SEED).fit(train, label_train)
    print(clf)
    # Check accuracy but this is based on the same data we used for training
    # Use classifier to train and test
    print('Result with NaiveBeys =*=*=*=*=')
    train_acc,train_precision,train_recall,train_f1,train_auc = predict_process(train,label_train,clf)
    test_acc,test_precision,test_recall,test_f1,test_auc = predict_process(test,label_test,clf)
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
    ]
    result = pd.DataFrame(pd.Series({
      'seed':GROUP,
      'classifier': classifier,
      'feature': feature,
      'numk': numk,
      'param':param_max,
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
    }))
    return result
  return 0