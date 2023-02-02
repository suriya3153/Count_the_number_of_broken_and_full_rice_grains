import cv2
import numpy as np
from keras.models import model_from_json
import pading_the_grain
import pandas as pd

grain_dict={0:"brokend",1:"full"}

json_file=open("model_js/rice_grain_model.json")
loaded_model_json=json_file.read()
json_file.close()
rice_clf_model=model_from_json(loaded_model_json)

rice_clf_model.load_weights("model_js/riceclf.h5")
print("model loaded")
# Load image
listcountofrice=[]
listcountofbrokenrice=[]
name=[]
for valuei in range(1,6):
    img = cv2.imread("data/test/image_{}.jpg".format(valuei))
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

            try:
                crop_img = pading_the_grain.pad_image(crop_img, 128, 0)
                croped_img = np.expand_dims(cv2.resize(crop_img,(128,128),cv2.INTER_LINEAR), axis=0)
                rice_pred=rice_clf_model.predict(croped_img)
                maxindex=int(np.argmax(rice_pred))
                cv2.putText(img,grain_dict[maxindex],(x+5,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
                if maxindex==0:
                    countofbrokenrice= countofbrokenrice +1
                countofrice = countofrice + 1

            except:
                pass
    listcountofrice.append(countofrice)
    listcountofbrokenrice.append(countofbrokenrice)
    name.append("image_{}.jpg".format(valuei))
df=pd.DataFrame({"file_name":name,"total_rice_grain":listcountofrice,"total_broken_rice_grain":listcountofbrokenrice})
df.to_csv("_submission_.csv")