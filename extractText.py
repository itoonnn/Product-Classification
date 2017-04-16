import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.utils import lemmatize
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.doc2vec import LabeledSentence,TaggedDocument
from gensim.models import Doc2Vec,Phrases
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora
import scipy
from random import shuffle
import os,sys,time

def bigram_model(documents):

  sentences = [[word for word in document.split() if word not in STOPWORDS and len(word)>1] for document in documents]
  bigram_transformer = Phrases(sentences)
  list_word = [bigram_transformer[sentence] for sentence in sentences]
  dictionary = corpora.Dictionary(list_word)
  corpus = [dictionary.doc2bow(sent) for sent in list_word]
  return corpus
def extract_tfid(doc):
  # Remove numbers in product name
  doc = doc.str.replace("[^a-zA-Z]", " ")
  doc = doc.str.lower()
  doc = doc.values.astype('U')
  # doc = doc.str.strip(" ")

  # vectorizer = TfidfVectorizer(ngram_range=(1,2),stop_words={'english'},min_df=0.001,max_df=1.0)
  # print(doc)
  # x = vectorizer.fit_transform(doc)

  corpus = bigram_model(doc)
  tfidf = TfidfModel(corpus)
  print("\n")
  print(tfidf)
  x = [tfidf[corp] for corp in corpus]

  return x,tfidf

def extract_w2v(doc,label,model_name="default",epochs=20):
  documents = doc
  sentences = [[word for word in document.split() if word not in STOPWORDS and len(word)>1] for document in documents]
  bigram_transformer = Phrases(sentences)
  documents = [LabeledSentence(words = bigram_transformer[sentences[i]], tags =["sent_"+str(label[i])]) for i in range(len(label))]
  nmax = 0
  nmin = 999999
  for sent in sentences:
    lsent = len(bigram_transformer[sent])
    nmax = max(nmax,lsent)
    nmin = min(nmin,lsent)
  print("MAX : ",nmax)
  print("MIN : ",nmin)
  model = gensim.models.doc2vec.Doc2Vec(dm=0, # DBOW refer to paper
          hs=1,  # soft max refer to paper
          size=10, # 100-300 is common
          sample=1e-5, # useful 
          alpha=0.01, # refer to paper
          min_alpha=1e-4, # refer to paper
          negative=5, #5-20 
          seed=2000,
          window=5, # good in 5-12 / if doc has < window-1 refer to NULL
          min_count=1)
  # build model
  model.build_vocab(documents)

  # train model
  training_data = list(documents)
  t = time.time()
  # model.train(training_data)
  # print(time.time()-t)
  ## fix alpha
  alpha_decrease = (model.alpha-model.min_alpha)/epochs
  for epoch in range(epochs):
    print("EPOCH : ",epoch,model.alpha,model.min_alpha)
    shuffle(training_data)
    model.train(training_data)
    model.alpha -= alpha_decrease  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
  print(time.time()-t)

  mname = model_name+".model"
  model.save(mname)
  model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
  return mname,documents

def extractTextFeature(data,label=[],opt="tfid",split=False,random_state = 2000,save=False,GROUP = 0,store= "coldstorage"):

  x = data
  if(not split):
    if(opt=="tfid"):
      x,vectorizer = extract_tfid(x)
    elif(opt=="w2v"):
      x,token = extract_w2v(x,label,model_name=store+"_"+str(GROUP))
      # return x,token
  else:
    y = label
    #### split train test
    SSK = StratifiedKFold(n_splits=10,random_state=random_state)
    INDEX = []
    for train_index, test_index in SSK.split(x,y):
      INDEX.append({'train':train_index,'test':test_index})
    train = x[INDEX[GROUP]['train']]
    test = x[INDEX[GROUP]['test']]
    label_train = y[INDEX[GROUP]['train']]
    label_test = y[INDEX[GROUP]['test']]
    if(opt=="tfid"):
      train,vectorizer = extract_tfid(train)
      # test = vectorizer.transform(test.values.astype('U'))
      test = bigram_model(test)
      test = vectorizer[test]
      print(np.shape(train))
      print(np.shape(test))
    elif(opt=="w2v"):
      mname,train_token = extract_w2v(train,train.index,model_name=store+"_"+str(GROUP))
      for i in train.index:
        vectorizer = Doc2Vec.load(mname)
        train_list = vectorizer.docvecs['sent_'+str(i)]
      # print(vectorizer.docvecs['sent_0']) # FAIL : train hasn't sent_0 
      # print(vectorizer.docvecs['sent_1']) # SUCCESS
      documents = test
      sentences = [[word for word in document.split() if word not in STOPWORDS and len(word)>1] for document in documents]
      bigram_transformer = Phrases(sentences)
      test_token = [TaggedDocument(words = bigram_transformer[sentences[i]], tags =[i]) for i in range(len(sentences))]
      for sentence in test_token:
        vectorizer = Doc2Vec.load(mname)
        test = vectorizer.infer_vector(sentence[0])
    
    # train = train.toarray()
    # test = test.toarray()
    # label_train = np.array(label_train)
    # label_test = np.array(label_test)
    # print(train)
    print(train)
    if(save):
      np.savetxt("feature_"+store+"_"+opt+"_train_"+str(GROUP)+".csv",train,delimiter=',')
      np.savetxt("feature_"+store+"_"+opt+"_test_"+str(GROUP)+".csv",test,delimiter=',')
      np.savetxt("label_"+store+"_"+opt+"_train_"+str(GROUP)+".csv",label_train)
      np.savetxt("label_"+store+"_"+opt+"_test_"+str(GROUP)+".csv",label_test)

    return train,test,label_train,label_test

# # # Specify input csv file
# print("file")
# print("coldstorage_path.csv == 1")
# print("giant_path.csv == 2")
# print("redmart_path.csv == 3")
# input_file = input()
# input_file = int(input_file)
# if input_file == 1:
#   input_file = "coldstorage_path.csv"
# elif input_file == 2:
#   input_file = "giant_path.csv"
# elif input_file == 3:
#   input_file = "redmart_path.csv"
# img_root = input_file.replace("_path.csv","")+"_img"
# print("SEED")
# SEED = 2000
# GROUP = int(input())
# print(input_file)


# df = pd.read_csv(input_file, header = 0)
# # Subset dataframe to just columns category_path and name
# df = df.loc[:,['category_path','name']]
# # Make a duplicate of input df
# df_original=df
# df_dedup=df.drop_duplicates(subset='name')
# # print(len(np.unique(df_dedup['name'])))
# df=df_dedup
# #drop paths that have 1 member
# df_count = df.groupby(['category_path']).count()
# df_count = df_count[df_count == 1]
# df_count = df_count.dropna()
# df = df.loc[~df['category_path'].isin(list(df_count.index))]
# df = df.reset_index(drop=True)
# df['name'] = df['name'].str.replace("[^a-zA-Z]", " ")
# df['name'] = df['name'].str.lower()

# print("Uniqued df by name : "+str(len(df['name'])))
# train,test,label_train,label_test = extractTextFeature(df['name'],label=df['category_path'],opt='tfid',split=True,fname=input_file)
# print(train)
# print(np.shape(train))
# print(token[0][0])
# model_name = x
# x = Doc2Vec.load(model_name)
# v2=x.infer_vector(token[0][0])
# x = Doc2Vec.load(model_name)
# v3=x.infer_vector(token[0][0])
# print(x.docvecs[0])
# print(x.docvecs[token[0][1]])
# print(x.docvecs['sent_0'])
# for key in x.docvecs:
#   print(key)

