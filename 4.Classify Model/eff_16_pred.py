import tensorflow.keras.backend as K
#from tensorflow.keras.models import load_model

K.clear_session()
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import *
from efficientnet.tfkeras import EfficientNetB3
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import utils
from keras import layers
import efficientnet.keras as efn
from keras.models import load_model
from sklearn.metrics import accuracy_score

model_best = load_model("/home/aigpuserver/efficient_simplekeras/eff_c16_best_model_4class.hdf5") 


from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

num_of_class = 4 

t_dict = {}
true_count = 0
false_count = 0
csv_list = []

def predict_class(model, images, show = True):
  #c_list=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18",]
  c_list = []
  global true_count
  global false_count
  global csv_list

  for c in range(0, num_of_class):
     c_list.append(str(c))
  c_list.sort()
  for img2 in images:
    img = image.load_img(img2, target_size=(299, 299))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.                                      

    pred = model.predict(img)
    print(pred)
    index = np.argmax(pred)
    print(index)
    pred0 = pred[0]
    pred_str = " ".join(["%.2f" % p  for p in pred0])
    #final_path = dst_folder3 +c_list[index] +"/"+ os.path.basename(img2)
    #print(final_path)
    #copyfile(img2,final_path)
    value = ()
    if int(t_dict[img2]) == int(c_list[index]):
        true_count = true_count + 1
        value = (img2,int(c_list[index]),int(t_dict[img2]),"T", pred_str)
    else:
        false_count = false_count + 1
        value = (img2,int(c_list[index]),int(t_dict[img2]),"F", pred_str)
    
    csv_list.append(value) 
    #food_list.sort()
    #pred_value = food_list[index]
    #print(food_list)
    #if show:
    #    plt.imshow(img[0])                           
    #    plt.axis('off')
    #    plt.title(pred_value)
    #    plt.show()

images = []
#images.append('samosa.jpg')
#images.append('p.jpeg')
#images.append('omelette.jpg')
#predict_class(model_best, images, True)

folder_name = "/home/aigpuserver/img_cls/c_data3/test/"

for c in range(0, num_of_class):
   
   final_ff = folder_name + str(c)
   file_list = os.listdir(final_ff)
  
   for f in file_list:
     img_p = final_ff + "/" + f
     t_dict[img_p] = int(c)
     images.append(img_p)

images.sort()
predict_class(model_best, images, True)

print("true count >", true_count)
print("false count >", false_count)

acc = float(true_count/(true_count+false_count))

print("accuracy,,,>", acc)

col_name = ["img_name","prediction", "True Value", "Result","softmax_values"]

csv_df = pd.DataFrame(csv_list, columns=col_name)

csv_df.to_csv("eff_16_final.csv", index=None)
