import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, filters, exposure, img_as_float, morphology
from skimage.color import rgb2gray
from skimage.feature import canny
from scipy import ndimage as ndi
import os,sys,cv2

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
  img_r = np.histogram(img_r, bins = np.arange(0,16))
  img_g = np.histogram(img_g, bins = np.arange(0,16))
  img_b = np.histogram(img_b, bins = np.arange(0,16))
  edges = np.histogram(edges, bins = np.arange(0,16))  
  shape = np.histogram(img_cleaned, bins = np.arange(0,16))  
  # print(edges)
  # plt.plot(img_r[0][:-1],img_r[0],color='r')
  # plt.plot(img_g[1][:-1],img_g[0],color='g')
  # plt.plot(img_b[1][:-1],img_b[0],color='b')
  # plt.show()
  ####### 
  vectorImg = np.array(img_r[0])
  vectorImg = np.append(vectorImg, img_g[0])
  vectorImg = np.append(vectorImg, img_b[0])
  vectorImg = np.append(vectorImg, edges[0])
  # vectorImg = np.append(vectorImg, shape[0])

  return vectorImg

def extractImage_sift(PATH):
  img = cv2.imread(PATH)
  gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  
  sift = cv2.xfeatures2d.SIFT_create()
  kp = sift.detect(gray,None)
  
  img=cv2.drawKeypoints(gray,kp)
  
  cv2.imwrite('sift_keypoints.jpg',img)
  return(img)

feature = extractImage_contextual("coldstorage_img/00a3918b5a5df518dc9379d94a7407b4.jpg")
# # feature = extractImage_sift("giant_img/00eab77ac36f6a4f77e1be12124e14ab.jpg")
print(np.shape(feature)) 
# print(feature)
