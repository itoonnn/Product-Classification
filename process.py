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

def naivebeys_process(SEED,GROUP,START,END,train,test,label_train,label_test):
  print("\nMultinomialNB")
  END += 1
  fname = "crossval_nb_result.csv"
  #Tune Alpha
  alpha = 0.0
  parameter_all = np.array([GRID_FN(i) for i in range(-5,6)])
  print("All param")
  print(parameter_all)
  columns = ['seed','alpha','score','std']
  df = pd.DataFrame(columns = columns)
  parameter_exist = []
  if(not os.path.isfile(fname)):
    df.to_csv(fname)
    print("create ",fname)
  else:
    data = df.from_csv(fname)
    print("Exist param")
    parameter_exist = np.array(data['alpha'])
    print(parameter_exist)
  #tune parameter
  for power in range(START,END,1):
    alpha = GRID_FN(power)
    if(alpha not in parameter_exist):
      #cross validation
      clf = MultinomialNB(alpha=alpha)
      scores = cross_validation.cross_val_score(clf,train,label_train,cv=10,scoring=roc_auc_my)
      score = round(scores.mean(),3)
      std = round(scores.std(),3)
      print(GROUP,alpha,score,std)
      df = pd.DataFrame([{'seed':GROUP,'alpha':alpha,'score':score,'std':std}],columns = columns)
      ### write file
      with open(fname, 'a') as f:
        df.to_csv(f, header=False)
      
  data = df.from_csv(fname)
  parameter_exist = np.array(data['alpha'])
  if np.array_equal(parameter_all, parameter_exist):
    print("Complete all parameter =*=*=*=")
    rec_max = data.sort(['score','std'],ascending=[0,1]).head(1)
    alpha_max = float(rec_max['alpha'])

    ### prediction start ###
    ## classified training
    clf = MultinomialNB(alpha=alpha_max).fit(train, label_train)
    print(clf)

    # Check accuracy but this is based on the same data we used for training
    # Use classifier to train and test
    print('Result with NaiveBeys =*=*=*=*=')
    train_acc,train_precision,train_recall,train_f1,train_auc = predict_process(train,label_train,clf)
    test_acc,test_precision,test_recall,test_f1,test_auc = predict_process(test,label_test,clf)

    columns = [
      'seed',
      'alpha',
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
      'alpha':alpha_max,
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


def neuralnetwork_process(SEED,GROUP,train,test,label_train,label_test):
  print("\nNeural Networks")
  fname = "crossval_nn_result.csv"
  #Tune Alpha
  layer = 3
  parameter_all = [(i) for i in range(10,110,10)]+[(i,i) for i in range(10,110,10)]+[(i,i,i) for i in range(10,110,10)]
  print("All param")
  print(parameter_all)
  columns = ['seed','hd_node','score','std']

  df = pd.DataFrame(columns = columns)
  parameter_exist = []
  if(not os.path.isfile(fname)):
    df.to_csv(fname)
    print("create ",fname)
  else:
    data = df.from_csv(fname)
    print("Exist param")

    parameter_exist = list_str2num(data['hd_node'])
    print(parameter_exist)
  #tune parameter
  for param in parameter_all:
    if(param not in parameter_exist):
      #cross validation
      clf = MLPClassifier(hidden_layer_sizes=param,random_state=SEED)
      scores = cross_validation.cross_val_score(clf,train,label_train,cv=10,scoring=roc_auc_my)
      score = round(scores.mean(),3)
      std = round(scores.std(),3)
      print(GROUP,param,score,std)
      df = pd.DataFrame([{'seed':GROUP,'hd_node':param,'score':score,'std':std}],columns = columns)
      ### write file
      with open(fname, 'a') as f:
        df.to_csv(f, header=False)
        print("write\n",df)
      
  data = df.from_csv(fname)
  parameter_exist = list_str2num(data['hd_node'])
  parameter_exist = [ tuple(int(j) for j in i ) if type(i)==tuple else int(i) for i in parameter_exist]
  if np.array_equal(parameter_all, parameter_exist) or parameter_all == parameter_exist:
    print("Complete all parameter =*=*=*=")
    rec_max = data.sort(['score','std'],ascending=[0,1]).head(1)
    param_max = str2int(rec_max['hd_node'][0])

    ### prediction start ###
    ## classified training
    clf = MLPClassifier(hidden_layer_sizes=param_max,random_state=SEED).fit(train, label_train)

    print(clf)

    # Check accuracy but this is based on the same data we used for training
    # Use classifier to train and test
    print('Result with NaiveBeys =*=*=*=*=')
    train_acc,train_precision,train_recall,train_f1,train_auc = predict_process(train,label_train,clf)
    test_acc,test_precision,test_recall,test_f1,test_auc = predict_process(test,label_test,clf)

    columns = [
      'seed',
      'hd_node',
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
      'hd_node':param_max,
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

def svm_process(SEED,GROUP,train,test,label_train,label_test,kernel='linear'):
  print("\nNeural Networks")
  fname = "crossval_svm_"+kernel+"_result.csv"
  #Tune Alpha
  layer = 3
  if(kernel == 'linear'):
    parameter_all = [GRID_FN(i) for i in range(-5,6)]
  elif(kernel == 'rbf'):
    parameter_all = [(GRID_FN(j),GRID_FN(i)) for j in range(-5,6) for i in range(-5,6)]
  elif(kernel == 'poly'):
    parameter_all = [(j,GRID_FN(i)) for j in range(1,6) for i in range(-5,6)]
  columns = ['seed','param','score','std']
  print(parameter_all)
  
  df = pd.DataFrame(columns = columns)
  parameter_exist = []
  if(not os.path.isfile(fname)):
    df.to_csv(fname)
    print("create ",fname)
  else:
    data = df.from_csv(fname)
    print("Exist param")
    print(data['param'])
    parameter_exist = list_str2num(data['param'])
    print(parameter_exist)
  #tune parameter
  for param in parameter_all:
    if(param not in parameter_exist):
      #cross validation
      if(kernel=="linear"):
        clf = SVC(C=param,kernel=kernel,probability=True,random_state=SEED)
      elif(kernel=="rbf"):
        clf = SVC(C=param[1],gamma=param[0],kernel=kernel,probability=True,random_state=SEED)
      elif(kernel=="poly"):
        clf = SVC(C=param[1],degree=param[0],kernel=kernel,probability=True,random_state=SEED)
      scores = cross_validation.cross_val_score(clf,train,label_train,cv=10,scoring=roc_auc_my)
      score = round(scores.mean(),3)
      std = round(scores.std(),3)
      print(GROUP,param,score,std)
      df = pd.DataFrame([{'seed':GROUP,'param':param,'score':score,'std':std}],columns = columns)

      ### write file
      with open(fname, 'a') as f:
        df.to_csv(f, header=False)
        print("write ",df)
      
  data = df.from_csv(fname)
  parameter_exist = np.array(data['hd_node'])
  if np.array_equal(parameter_all, parameter_exist):
    print("Complete all parameter =*=*=*=")
    rec_max = data.sort(['score','std'],ascending=[0,1]).head(1)
    param_max = float(rec_max['hd_node'])

    ### prediction start ###
    ## classified training
    clf = MLPClassifier(hidden_layer_sizes=param_max,random_state=SEED).fit(train, label_train)
    print(clf)

    # Check accuracy but this is based on the same data we used for training
    # Use classifier to train and test
    print('Result with NaiveBeys =*=*=*=*=')
    train_acc,train_precision,train_recall,train_f1,train_auc = predict_process(train,label_train,clf)
    test_acc,test_precision,test_recall,test_f1,test_auc = predict_process(test,label_test,clf)

    if(kernel == 'linear'):
      columns = [
        'seed',
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
  
def logistic_process(SEED,GROUP,START,END,train,test,label_train,label_test):
  print("\nLogistic Regression")
  END += 1
  fname = "crossval_lr_result.csv"
  #Tune Alpha
  C = 0.0
  parameter_all = np.array([GRID_FN(i) for i in range(-5,6)])
  print("All param")
  print(parameter_all)
  columns = ['seed','C','score','std']
  df = pd.DataFrame(columns = columns)
  parameter_exist = []
  if(not os.path.isfile(fname)):
    df.to_csv(fname)
    print("create ",fname)
  else:
    data = df.from_csv(fname)
    print("Exist param")
    parameter_exist = np.array(data['C'])
    print(parameter_exist)
  #tune parameter
  for power in range(START,END,1):
    C = GRID_FN(power)
    if(C not in parameter_exist):
      #cross validation
      clf = LogisticRegression(random_state=SEED,C=C,n_jobs=-1,multi_class='multinomial',solver='sag')
      scores = cross_validation.cross_val_score(clf,train,label_train,cv=10,scoring=roc_auc_my)
      score = round(scores.mean(),3)
      std = round(scores.std(),3)
      print(GROUP,C,score,std)
      df = pd.DataFrame([{'seed':GROUP,'C':C,'score':score,'std':std}],columns = columns)
      ### write file
      with open(fname, 'a') as f:
        df.to_csv(f, header=False)
      
  data = df.from_csv(fname)
  parameter_exist = np.array(data['C'])
  if np.array_equal(parameter_all, parameter_exist):
    print("Complete all parameter =*=*=*=")
    rec_max = data.sort(['score','std'],ascending=[0,1]).head(1)
    C_max = float(rec_max['C'])

    ### prediction start ###
    ## classified training
    clf = LogisticRegression(random_state=SEED,C=C_max,n_jobs=-1,multi_class='multinomial',solver='sag').fit(train, label_train)
    print(clf)

    # Check accuracy but this is based on the same data we used for training
    # Use classifier to train and test
    print('Result with NaiveBeys =*=*=*=*=')
    train_acc,train_precision,train_recall,train_f1,train_auc = predict_process(train,label_train,clf)
    test_acc,test_precision,test_recall,test_f1,test_auc = predict_process(test,label_test,clf)

    columns = [
      'seed',
      'C',
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
      'C':C_max,
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