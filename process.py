import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
from sklearn.preprocessing import LabelEncoder,FunctionTransformer,Normalizer,label_binarize,MultiLabelBinarizer
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
def getResult(predictions,labelData,probas_):
  pos_label = max(labelData)
  acc = accuracy_score(labelData,predictions)
  (precision,recall,fbeta,support) = precision_recall_fscore_support(labelData,predictions,pos_label=pos_label,average="weighted")
  # report = classification_report(labelData,predictions)
  auc_score = roc_auc_fixed(labelData,predictions)
  return acc,precision,recall,fbeta,auc_score
def naivebeys_process(SEED,GROUP,START,END,train,test,label_train,label_test):
  print("\nMultinomialNB")
  END += 1
  fname = "crossval_nb_result_"+str(GROUP)+".csv"
  #Tune Alpha
  alpha = 0.0
  scores_mean_nv = []
  scores_std_nv = []
  alphas = ()

  dtype = [('ngram',int),('alpha',float),('score',float),('std',float)]
  scores = pd.DataFrame(dtype = dtype)
  if(os.path.isfile(fname) ):
    scores.to_csv(fname)
  #tune parameter
  for power in range(START,END,1):
    alpha = GRID_FN(power)
    alphas+=(alpha,)
    #cross validation
    clf = MultinomialNB(alpha=alpha)
    scores = cross_validation.cross_val_score(clf,train,label_train,cv=10,scoring=roc_auc_my)
    # print scores
    print(ngram,alpha,scores.mean(), scores.std())
    scores_mean_nv.append(scores.mean())
    scores_std_nv.append(scores.std())
    ### write file
    

    ###

  # scores_mean_nv = np.array(scores_mean_nv)
  # scores_std_nv = np.array(scores_std_nv)
  
  # scores_zip = zip(ngrams,alphas,scores_mean_nv,scores_std_nv)
  # scores_list = pd.DataFrame(np.array(scores_zip,dtype=dtype))

  # scores_max_nv = scores_list.sort(['score','std'],ascending=[0,1]).head(1)
  # max_alpha_nv = scores_max_nv.get_value(scores_max_nv.index.values[0],'alpha')

  # print "NV score : "
  # print scores_max_nv

  # ### prediction start ###
  # ## classified training
  # clf = MultinomialNB(alpha=max_alpha_nv).fit(train, label_train)
  # print(clf)

  # # Check accuracy but this is based on the same data we used for training
  # # Use classifier to train and test
  # print 'Result of Generation ',gen,' with NaiveBeys'
  # train_acc,train_precision,train_recall,train_f1,train_auc = predict_process("train",train,label_train,clf)
  # test_acc,test_precision,test_recall,test_f1,test_auc = predict_process("test",test,label_test,clf)

  # print "ngram : ",ngram
  # print "alpha : ",max_alpha_nv
  # dtype = [('ngram',int),('alpha',float),('train_acc',float),('train_precision',float),('train_recall',float),('train_f1',float),('train_auc',float),('test_acc',float),('test_precision',float),('test_recall',float),('test_f1',float),('test_auc',float)]
  # scores_result = pd.DataFrame(np.array([(
  #   ngram,
  #   max_alpha_nv,
  #   train_acc,
  #   train_precision,
  #   train_recall,
  #   train_f1,
  #   train_auc,
  #   test_acc,
  #   test_precision,
  #   test_recall,
  #   test_f1,
  #   test_auc
  # )],dtype=dtype))
  # return scores_list,scores_result,train_report,test_report