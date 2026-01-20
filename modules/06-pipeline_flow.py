import cv2
import joblib

haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
mean = joblib.load('./model/mean.joblib')
svc = joblib.load('./model/svc_model.joblib')
pca = joblib.load('./model/pca_50.joblib')

gender = ['Male', 'Female']
font = cv2.FONT_HERSHEY_COMPLEX

def pipeline_flow(img, color='rgb'):
    # convert into gray scale
    if color == 'bgr':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: # rgb
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # crop the face (using harr cascade classifier)
    faces = haar.detectMultiScale(gray, 1.5, 3)
    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) # drawing rectangle
        roi = gray[y:y+h, x:x+w] # actual cropping
        # normalization (0-1)
        roi = roi / 255.0 # acording to what we did in a previous module
        # resize image (100,100)
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100,100), cv2.INTER_AREA)
        else: # roi.shape[1] < 100
            roi_resize = cv2.resize(roi, (100,100), cv2.INTER_CUBIC)
        # flattering (1x10,000)
        roi_reshape = roi_resize.reshape(1, 10000)
        # substract with mean
        roi_mean = roi_reshape - mean
        # get eigen image
        eigen_image = pca.transform(roi_mean)
        # pass to ml model
        results = svc.predict_proba(eigen_image)[0]
        # results = svc.predict(eigen_image)[0]
        predict = results.argmax() # 0 or 1
        score = results[predict]

        text = '%s - %0.2f'%(gender[predict], score)
        cv2.putText(img, text, (x,y), font, 1, (0,255,0), 2)

    return img

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_path = './data/male.jpg'
    # read image
    img = cv2.imread(test_path) # must be cv2 to draw the rectangle
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert color

    img = pipeline_flow(img)
    plt.imshow(img)
    plt.show()