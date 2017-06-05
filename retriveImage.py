import numpy as np
import pandas as pd
import os,sys
import urllib.request as ur

FILE = "redmart.csv"
FOLDER_NAME = "redmart_img"

try:
  os.mkdir(FOLDER_NAME)
except WindowsError:
  pass

def retreiveImg(url,folder):
  fname = url.split("/")[-1]
  f = open(folder+"/"+fname,'wb')
  f.write(ur.urlopen(url).read())
  f.close()

dataFile = pd.read_csv(FILE, header = 0)
#add new column
dataFile['img_file'] = dataFile['img_url'].str.split("/").str[-1]
dataFile.to_csv(FILE)
#drop duplicate
dataFile = dataFile.loc[:,['img_url','name']]
dataFile_dedup=dataFile.drop_duplicates(subset='name')
dataFile=dataFile_dedup
dataFile = dataFile.reset_index(drop=True)
# print(dataFile['img_url'][0].split("/")[-1])
for imgurl in dataFile['img_url']:
  print(imgurl)
  try:
    if(os.path.isfile(FOLDER_NAME+"/"+fname)):
      print("EXISTING")
    else:
      retreiveImg(imgurl,FOLDER_NAME)
      print("DOWNLOAD...")
  except:
    print("ERROR")
    pass