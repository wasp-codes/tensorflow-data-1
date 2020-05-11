import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# load model
#model.load_weights('asphaltCrackData.h5')
model = load_model('asphaltCrackData.h5')


test_image = image.load_img('t.jpg', target_size = (227, 227))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
#train_generator.class_indices
if result[0][0] >= 0.5:
  prediction = 'UNCRACKED'
else:
  prediction = 'CRACKED'

#print(result[0][0])
print(prediction)