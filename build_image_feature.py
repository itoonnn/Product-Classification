from extractImage import *
from sklearn.preprocessing import LabelEncoder

input_file = "redmart_path.csv"
store = "redmart"
img_root = store+"_img"
df = pd.read_csv(input_file, header = 0)
### Preprocessing  start ###
###
# Subset dataframe to just columns category_path and name
df = df.loc[:,['category_path','name','img_file']]
# Make a duplicate of input df
df_original=df
df_dedup=df.drop_duplicates(subset='name')
# print(len(np.unique(df_dedup['name'])))
df=df_dedup
#drop paths that have 1 member
df_count = df.groupby(['category_path']).count()
df_count = df_count[df_count == 1]
df_count = df_count.dropna()
df = df.loc[~df['category_path'].isin(list(df_count.index))]
df = df.reset_index(drop=True)
print("Uniqued df by name : "+str(len(df['name'])))
####### label encoder
number = LabelEncoder()
y = df['category_path']
y = y.astype(str) 
y = number.fit_transform(y)
print(np.shape(df['img_file']))
for i in range(0,10):
  x = extractImageFeature(df['img_file'],img_root,label=y,opt="sift",split=True,random_state=2000,save=True,GROUP=i)