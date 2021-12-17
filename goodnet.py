from tensorflow import keras
import sys
import numpy as np
from keras.models import load_model
from matplotlib import image

image_path = str(sys.argv[1])
x = str(sys.argv[2])
bd_model = load_model('/models/bd_net.h5')
if x == '10':
    pruned_model = load_model('/models/pruned_10.h5')
elif x == '4':
    pruned_model = load_model('/models/pruned_4.h5')
else:
    pruned_model = load_model('/models/pruned_2.h5')

image = image.imread(image_path)

bd_label = np.argmax(bd_model.predict(image[None,...]), axis=1)
pr_label = np.argmax(pruned_model.predict(image[None,...]), axis=1)

    
if bd_label[0] == pr_label[0]:
    print("The predicted class id is ", pr_label[0])
else:
    print("The predicted class id is 1283.")
