import cv2
import numpy as np
from keras.models import model_from_json
import pading_the_grain

grain_dict={0:"broken",1:"full"}

json_file=open("model_js/rice_grain_model.json")
loaded_model_json=json_file.read()
json_file.close()
rice_clf_model=model_from_json(loaded_model_json)

rice_clf_model.load_weights("model_js/riceclf.h5")
print("model loaded")
# Load image
img = cv2.imread("data/test/image_2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bounding_rects = [cv2.boundingRect(c) for c in contours]

# Crop and save each rice grain
countofrice=0
countofbrokenrice=0


for x, y, w, h in bounding_rects:
    crop_img = img[y:y + h, x:x + w]
    height, width = crop_img.shape[:2]
    total_pixels = height * width
    if total_pixels >= 1000 and total_pixels <= 4000:
        cv2.rectangle(img,(x,y-50),(x+w,y+h+10),(0,255,0),4)
        crop_img = img[y:y+h, x:x+w]
        crop_img = pading_the_grain.pad_image(crop_img, 128, 0)
        try:
            croped_img = np.expand_dims(cv2.resize(crop_img,(128,128),cv2.INTER_LINEAR), axis=0)
            rice_pred=rice_clf_model.predict(croped_img)
            maxindex=int(np.argmax(rice_pred))

            if grain_dict[maxindex]=="broken":
                cv2.putText(img, grain_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)
                countofbrokenrice=countofbrokenrice+1
            else:
                cv2.putText(img, grain_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
            countofrice = countofrice + 1
        except:
            pass
print(countofrice)
print(countofbrokenrice)
img=cv2.resize(img,(1280,720))
cv2.imshow("pred",img)
cv2.waitKey(0)