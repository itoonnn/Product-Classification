import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
from sklearn.preprocessing import LabelEncoder,FunctionTransformer,Normalizer,label_binarize,MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, precision_score ,recall_score, roc_curve, auc, roc_auc_score,make_scorer,precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from extractImage import *

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

TEST_SIZE = 0.2
# Specify input csv file
print("file")
print("coldstorage_path.csv == 1")
print("giant_path.csv == 2")
print("redmart_path.csv == 3")
input_file = input()
input_file = int(input_file)
if input_file == 1:
  input_file = "coldstorage_path.csv"
elif input_file == 2:
  input_file = "giant_path.csv"
elif input_file == 3:
  input_file = "redmart_path.csv"
store = input_file.replace("_path.csv","")
img_root = store+"_img"
print("Opertion contextural,sift,surf,orb")
opt = input()
print("Number of K cluster")
nfeature = int(input())
print("Number of C in SVM")
C = float(input())
print("SEED")
SEED = 2000
GROUP = int(input())
print(input_file)
df = pd.read_csv(input_file, header = 0)


x = np.loadtxt(store+"_feature_"+opt+"_"+str(GROUP)+".csv",delimiter=',')
y = np.loadtxt(store+"_label.csv",delimiter=',')


#### split train test
SSK = StratifiedKFold(n_splits=10,random_state=SEED)
INDEX = []
for train_index, test_index in SSK.split(x,y):
  INDEX.append({'train':train_index,'test':test_index})
train = x[INDEX[GROUP]['train']]
test = x[INDEX[GROUP]['test']]
label_train = y[INDEX[GROUP]['train']]
label_test = y[INDEX[GROUP]['test']]

# unique, counts = np.unique(label_train, return_counts=True)
# print(np.asarray((unique,counts)).T)

##### classification
print(train)
clf = SVC(C=C,probability=True,random_state=SEED).fit(train, label_train)
pred = clf.predict(train)
probas_ = clf.predict_proba(train)
acc,precision,recall,fbeta,auc_score = getResult(pred,label_train,probas_)
print("TRAIN RESULT")
print("accuracy :",acc)
print("precision :",precision)
print("recall :",recall)
print("f-score :",fbeta)
print("AUROC :",auc_score)