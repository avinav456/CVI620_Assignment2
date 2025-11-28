import cv2
import numpy as np
from keras.models import load_model
import joblib

image = cv2.imread("test_image.jpg")  
img = cv2.resize(image, (32, 32))
img = img/255
img = np.expand_dims(img, axis=0)

model = load_model("cat_dog_cnn_model.h5")
le = joblib.load("label_encoder.z") 

prediction = model.predict(img)
predicted_class = np.argmax(prediction)
predicted_label = le.inverse_transform([predicted_class])[0]

cv2.putText(image, predicted_label, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
print(predicted_label)
cv2.imshow('frame', image)
cv2.waitKey(0)
cv2.destroyAllWindows()