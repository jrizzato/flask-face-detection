import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from glob import glob

female = glob("./data/crop/female_crop/*.png")
male = glob("./data/crop/male_crop/*.png")

# print(female) # each position of the list is a string with the path of the image
# print(type(female))
# print(len(female))
# print(len(male))

path = female + male
# print(type(path)) # list

# --- get size of image ---
def get_size(path):
    img = Image.open(path)
    return img.size[0] # all are square images, so we take only first dimention

# create dataframe
df = pd.DataFrame(data=path, columns=["path"])
print(df.head()) # df only contains the path por now

df['size'] = df['path'].apply(get_size)

print(df.head())
print(type(df)) # path and size

# let`s perform some simple EDA
print(df.describe())

# print(type(df['size'].plot.box())) # <class 'matplotlib.axes._axes.Axes'>
plt.subplot(1,2,1)
df['size'].plot.box()

plt.subplot(1,2,2)
plt.hist(df['size'], bins=30)

plt.show()

# i will resize all images to 100x100 but before i will remove all images below 54x54
# df_new = df[df['size'] > 54]
# print(df_new.head())
# print(df_new.shape)

# add a column with gender based on the name of the file
string = df['path'][0]
print(string) # ./data/crop/female_crop\female_0.png
splited = string.split('_')
print(splited) # ['./data/crop/female', 'crop\\female', '0.png']
splited = string.split('_')[0]
print(splited) # ./data/crop/female
splited = string.split('_')[0].split('/')
print(splited) # ['.', 'data', 'crop', 'female']
splited = string.split('_')[0].split('/')[-1]
print(splited) # female
# so, based on the name of the file, we can know if it is an image of male o female

def gender(string):
    try:
        return string.split('_')[0].split('/')[-1]
    except Exception as e:
        print(f'ERROR: {e}')

df['gender'] = df['path'].apply(gender)
print(df.head()) # path, size and gender

plt.figure(figsize=(5,7))
print(df['gender'].value_counts(normalize=True))
df['gender'].value_counts(normalize=True).plot.bar()
plt.show()
# so the categories male or female are pretty balanced

def resize_img(path_to_resize):
    # this function resize and reshape de image
    # step 1:read image
    img = cv2.imread(path_to_resize)
    #step 2: convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # step 3: resize into 100 x 100
    size = gray.shape[0]
    if size >= 100: #shrink
        gray_rs = cv2.resize(gray, (100,100), cv2.INTER_AREA)
    else: # <100, enlarge
        gray_rs = cv2.resize(gray, (100,100), cv2.INTER_CUBIC)

    flat_img = gray_rs.flatten() # reshape from 2D to 1D
    
    return flat_img

print(len(resize_img(path[0]))) # 10,000

df['structure'] = df['path'].apply(resize_img)
print(df.head()) #path, size, gender and structure (pixel by pixel)

# so nowo we have a df that contains the path, size, gender and flatten structure.
# now lets create a new df 
df1 = df['structure']
print(df1.head()) #df2 contains only the strcutrue in one column
# expand df1 columns
df1 = df['structure'].apply(pd.Series)
print(df1.head()) # df1 now have 10,000 features (10,000 columns)

df2 = pd.concat((df['gender'], df1), axis=1) # axis=1 means columns
print(df2.head()) # df2 contains gender and features

plt.imshow(df2.loc[0][1:].values.reshape(100,100).astype('int'), cmap='gray')
plt.title('Label: ' + df2.loc[0]['gender'])
plt.show()

plt.imshow(df2.loc[5000][1:].values.reshape(100,100).astype('int'), cmap='gray')
plt.title('Label: ' + df2.loc[5000]['gender'])
plt.show()

import joblib
joblib.dump(df2, './data/df_images_100_100.joblib')