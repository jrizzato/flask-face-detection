from modules.config import app
import matplotlib.pyplot as plt
import cv2

img_plt = plt.imread('./data/test.jpg') # codificada como RGB
print(img_plt.shape) # 1200x1600x3
print("img_plt", type(img_plt)) # numpy.ndarray

img_cv2 = cv2.imread("./data/test.jpg") # codificada como BGR
print("img_cv2", type(img_cv2)) # numpy.ndarray

plt.imshow(img_plt) # los colores se ven normal
plt.show()

plt.imshow(img_cv2) # los colores se ven distorcionados
plt.show()

img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB) # convert color
plt.imshow(img_cv2_rgb)
plt.show()

cv2.imshow("Detect", img_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows
