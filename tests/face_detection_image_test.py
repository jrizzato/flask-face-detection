import cv2

# read image
img = cv2.imread("./data/male.jpg")
# convert into gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# apply haar cascade
# https://github.com/opencv/opencv/tree/master/data/haarcascades
haar = cv2.CascadeClassifier("./model/haarcascade_frontalface_default.xml")
faces = haar.detectMultiScale(gray, 1.3, 5)

print(faces) # [[154  94 261 261]]
cv2.rectangle(img, (154, 94), (154+261, 94+261), (0,255,0), 3) # (imagen, p1, p2, color, grosor)
cv2.imshow("Detect", img)
cv2.waitKey(0)
cv2.destroyAllWindows

# crop
face_crop = img[94:94+261, 154:154+261]
cv2.imshow("Crop", face_crop)
cv2.waitKey(0)
cv2.destroyAllWindows

# save
cv2.imwrite("./data/male_crop.png", face_crop)