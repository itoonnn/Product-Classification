import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from skimage import io, filters, exposure, img_as_float, morphology
from skimage.color import rgb2gray
from skimage.feature import canny
from sklearn.cluster import KMeans
from scipy import ndimage as ndi
from scipy.spatial import distance
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
  img = cv2.imread(PATH)
  gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  
  sift = cv2.xfeatures2d.SIFT_create()
  kp, des = sift.detectAndCompute(gray,None)
  
  return des

def extractImage_surf(PATH,threshold=400):
  img = cv2.imread(PATH)
  gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  
  surf = cv2.xfeatures2d.SURF_create(threshold)
  kp, des = surf.detectAndCompute(gray,None)
  return des

def extractImage_orb(PATH):
  img = cv2.imread(PATH)
  gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  
  orb = cv2.ORB_create(edgeThreshold=0,scoreType=cv2.ORB_FAST_SCORE)
  kp, des = orb.detectAndCompute(gray,None)
  return des

def extractImageFeature(data,img_root,opt='contextual',random_state = 2000):

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
  if(opt in ['sift','surf','orb']):
    for i in range(len(x)):
      if(np.shape(x[i])==()):
        x[i] = np.zeros(miss_shape)
      else:
        norm = Normalizer()
        x[i] = norm.fit_transform(x[i])
    x_cluster = np.vstack(x)
    print("START Clustering")
    # n_clusters = 1000
    n_clusters = math.floor(math.sqrt(len(x_cluster)))
    print("Total keypoint : ",len(x_cluster))
    print("Number of cluster : ",n_clusters)
    x_cluster = KMeans(n_clusters=n_clusters,random_state = random_state).fit(x_cluster)
    centroids = x_cluster.cluster_centers_
    labels = x_cluster.labels_
    print("EXTRACT HISTOGRAM")
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

