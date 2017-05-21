import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from skimage import io, filters, exposure, img_as_float, morphology
from skimage.color import rgb2gray
from skimage.feature import canny
from sklearn.cluster import MiniBatchKMeans
from scipy import ndimage as ndi
from scipy.spatial import distance
from sklearn.model_selection import StratifiedKFold
import os,sys,cv2,math

def extractImage_contextual(PATH):
  img = io.imread(PATH)
  img = img_as_float(img)
  ####### seperate channel color
  img_r = img[:,:,0]
  img_g = img[:,:,1]
  img_b = img[:,:,2]
  ####### gray scale
  img_gray = rgb2gray(img)
  ####### filter edges
  edges = canny(img_gray)
  # io.imshow(edges)
  # io.show()
  ####### shape
  fill_img = ndi.binary_fill_holes(edges)
  img_cleaned = morphology.remove_small_objects(fill_img,21)
  # io.imshow(img_cleaned)
  # io.show()
  ####### binarize with histogram
  
  img_r = np.histogram(np.reshape(img_r,-1),range=(0,1), bins = 16)
  img_g = np.histogram(np.reshape(img_g,-1),range=(0,1), bins = 16)
  img_b = np.histogram(np.reshape(img_b,-1),range=(0,1), bins = 16)
  img_color = exposure.histogram(img,nbins=16)
  edges = np.histogram(np.reshape(edges,-1),range=(0,1), bins = 16)  
  shape = np.histogram(img_cleaned, bins = 16)  
  ####### 
  vectorImg = np.array(img_color[0])
  # vectorImg = np.array(img_r[0])
  # vectorImg = np.append(vectorImg, img_g[0])
  # vectorImg = np.append(vectorImg, img_b[0])
  vectorImg = np.append(vectorImg, edges[0])
  # vectorImg = np.append(vectorImg, shape[0])

  return vectorImg

def extractImage_sift(PATH):
  img = cv2.imread(PATH,1)
  # gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  sift = cv2.xfeatures2d.SIFT_create()
  kp, des = sift.detectAndCompute(img,None)
  
  return des

def extractImage_surf(PATH,threshold=400):
  img = cv2.imread(PATH,1)
  # gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  surf = cv2.xfeatures2d.SURF_create(threshold)
  kp, des = surf.detectAndCompute(img,None)
  return des

def extractImage_orb(PATH):
  img = cv2.imread(PATH)
  gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  
  orb = cv2.ORB_create(edgeThreshold=0,scoreType=cv2.ORB_FAST_SCORE)
  kp, des = orb.detectAndCompute(gray,None)
  return des

def build_histrogram(x,centroids,labels,n_clusters):
  c = 0
  x_keypoint = x
  x = []
  nmin = 999999
  nmax = 0
  print(len(x_keypoint))
  for i in range(len(x_keypoint)):
    feature = np.zeros(n_clusters)
    nmax = len(x_keypoint[i]) if len(x_keypoint[i])> nmax else nmax
    nmin = len(x_keypoint[i]) if len(x_keypoint[i])< nmin else nmin
    for j in range(len(x_keypoint[i])):
      cluster = labels[c]
      sim = distance.euclidean(x_keypoint[i][j], centroids[cluster]) #find similarity betwee keypoint and centroid of cluster of this keypoint
      feature[cluster]+= sim
      c+=1
    x.append(feature)
  x = np.array(x)
  print("min ",nmin)
  print("max ",nmax)
  return x

def extractImageFeature(data,img_root,label=[],opt='contextual',random_state = 2000,split=False,save=False,GROUP = 0):
  store = img_root.replace("_img","")
  x = []
  miss_shape = 0
  i = 1
  for i,img in enumerate(data):
    # if(i%100==0):
    #   print(i)
    try:
      if( opt == 'contextual'):
        feature = np.array(extractImage_contextual(img_root+"/"+img))
      elif( opt == 'sift'):
        try:
          feature = np.array(extractImage_sift(img_root+"/"+img))
        except:
          feature = np.zeros(miss_shape)
      elif( opt == 'surf'):
        try:
          feature = np.array(extractImage_surf(img_root+"/"+img))
        except:
          feature = np.zeros(miss_shape)
      elif( opt == 'orb'):
        try:
          feature = np.array(extractImage_orb(img_root+"/"+img))
        except:
          feature = np.zeros(miss_shape)
      miss_shape = np.shape(feature) if np.shape(feature) != () else miss_shape
    except:
      feature = np.zeros(miss_shape)

    x.append(feature)
  x = np.array(x)
  if( not split ):
    if(opt in ['sift','surf','orb']):
      print(x)
      for i in range(len(x)):
        if(np.shape(x[i])==()):
          x[i] = np.zeros(miss_shape)
        else:
          norm = Normalizer()
          x[i] = norm.fit_transform(x[i])
      x_cluster = np.vstack(x)
      print("START Clustering")
      n_clusters = 10
      # n_clusters = math.floor(math.sqrt(len(x_cluster)))
      print("Total keypoint : ",len(x_cluster))
      print("Number of cluster : ",n_clusters)
      cluster = MiniBatchKMeans(n_clusters=n_clusters,random_state = random_state).fit(x_cluster)
      centroids = cluster.cluster_centers_
      labels = cluster.labels_
      print("EXTRACT HISTOGRAM")
      x = build_histrogram(x,centroids,labels,n_clusters)
    return x
  else:
    x = np.array(x)
    y = label[['top_name', 'second_name', 'third_name']].replace(99, 'NA')
    y = y[['top_name', 'second_name', 'third_name']]
    y['label'] = y['top_name'].astype(str).str.cat(y['second_name'].astype(str),sep="->")
    y['label'] = y['label'].astype(str).str.cat(y['third_name'].astype(str),sep="->")
    y = y['label']
    print(np.shape(y))
    print(np.shape(x))
    #### split train test
    SSK = StratifiedKFold(n_splits=10,random_state=random_state)
    INDEX = []
    for train_index, test_index in SSK.split(x,y):
      INDEX.append({'train':train_index,'test':test_index})
    train = x[INDEX[GROUP]['train']]
    test = x[INDEX[GROUP]['test']]
    label_train = label.loc[INDEX[GROUP]['train']]
    label_test = label.loc[INDEX[GROUP]['test']]
    print(np.shape(test))
    if(opt in ['sift','surf','orb']):
      for i in range(len(train)):
        if(np.shape(train[i])==()):
          train[i] = np.zeros(miss_shape)
        else:
          norm = Normalizer()
          train[i] = norm.fit_transform(train[i])
      for i in range(len(test)):
        if(np.shape(test[i])==()):
          test[i] = np.zeros(miss_shape)
        else:
          norm = Normalizer()
          test[i] = norm.fit_transform(test[i])

      train_cluster = np.vstack(train)
      test_cluster = np.vstack(test)
      print("START Clustering")
      # n_clusters = 1000 ###########################################################################
      # n_clusters = math.floor(math.sqrt(len(train_cluster)))
      n_clusters = math.floor(math.sqrt(len(train_cluster)/2)) ###############################################
      print("Total keypoint : ",len(train_cluster))
      print("Number of cluster : ",n_clusters)
      cluster = MiniBatchKMeans(n_clusters=n_clusters,random_state = random_state).fit(train_cluster)
      centroids = cluster.cluster_centers_
      train_cluster_labels = cluster.labels_
      test_cluster_labels = cluster.predict(test_cluster)
      print("EXTRACT HISTOGRAM")
      train = build_histrogram(train,centroids,train_cluster_labels,n_clusters)
      test = build_histrogram(test,centroids,test_cluster_labels,n_clusters)
      if(save):
        np.savetxt("../img_feature/sqrt(half(n))/feature_"+store+"_"+opt+"_train_"+str(GROUP)+".csv",train,delimiter=',')
        np.savetxt("../img_feature/sqrt(half(n))/feature_"+store+"_"+opt+"_test_"+str(GROUP)+".csv",test,delimiter=',')
        label_train.to_csv("../img_feature/sqrt(half(n))/label_"+store+"_"+opt+"_train_"+str(GROUP)+".csv",sep=",")
        label_test.to_csv("../img_feature/sqrt(half(n))/label_"+store+"_"+opt+"_test_"+str(GROUP)+".csv",sep=",")
      
    return train,test,label_train,label_test


# # # feature = extractImage_contextual("coldstorage_img/00a3918b5a5df518dc9379d94a7407b4.jpg")
# # # feature = extractImage_contextual("giant_img/00eab77ac36f6a4f77e1be12124e14ab.jpg")
# feature = extractImage_orb("giant_img/00eab77ac36f6a4f77e1be12124e14ab.jpg")
# # feature = extractImage_sift("coldstorage_img/fb89b62b794e9eaca4289ce7a028d948.jpg")
# print(np.shape(feature)) 
# print(feature)


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
# df = df.loc[:,['category_path','name','img_file']]
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
# print("Uniqued df by name : "+str(len(df['name'])))


# x = extractImageFeature(df['img_file'][:],img_root,opt='sift')
# # print(x)

