import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

female_path = glob("./data/female/*.jpg")
# print(female_path)
male_path = glob("./data/male/*.jpg")
# print(female_path)

print(len(female_path))
print(len(male_path))

# let's check an image
path = female_path[0]
img = cv2.imread(path)
cv2.imshow("female",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# let's convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# load haar cascade classifier
haar = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
faces = haar.detectMultiScale(gray, 1.5, 5)
# print(faces) # [[ 86  86 273 273]] 
for x,y,w,h in faces:
    cv2.rectangle(img, (x, y) , (x+w, y+h), (0, 255, 0), 2)

# Convertir de BGR a RGB para matplotlib
img_plt_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_plt_rgb)
plt.title("Detect")
plt.axis('on')  # para ver los ejes y corroborar las coordenadas de los putnos
plt.show()

crop_img = img[y:y+h, x:x+w]
cv2.imshow("Crop", crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save image
# cv2.imwrite("f_01.png", crop_img)

# apply to all the images
def extract_images(path, gender, i):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.5, 5)
    for x,y,w,h in faces:
        roi = img[y:y+h, x:x+w]
        if gender == 'female':
            cv2.imwrite(f'./data/crop/female_crop/{gender}_{i}.png', roi)
        elif gender == 'male':
            cv2.imwrite(f'./data/crop/male_crop/{gender}_{i}.png', roi)

# extract_images(female_path[0], 'female', 1)

# for i, path in enumerate(female_path):
#     try:
#         extract_images(path, 'female', i)
#         print(f'INFO: {i}/{len(female_path)} processed sucessfully')

#     except Exception as e:
#         print(f'ERROR: {e}')

for i, path in enumerate(male_path):
    try:
        extract_images(path, 'male', i)
        print(f'INFO: {i}/{len(male_path)} processed sucessfully')
        
    except Exception as e:
        print(f'ERROR: {e}')

