import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os

#image_path2 = "/home/joyhyuk/python/img_cls/data2/test/images/"
#label_path2 = "/home/joyhyuk/python/img_cls/data2/test/labels/"

#train/valid/test 바꿔가며 전부 만들기

dst_folder = "C:\Users\user\Desktop\modeling\daejang_data\full\train" #image list
dst_folder2 ="C:\Users\user\Desktop\modeling\daejang_data\full\train" #image list

num_of_class = 4 

def merge_image(back, front, x,y):
    # convert to rgba
    if back.shape[2] == 3:
        back = cv2.cvtColor(back, cv2.COLOR_BGR2BGRA)
    if front.shape[2] == 3:
        front = cv2.cvtColor(front, cv2.COLOR_BGR2BGRA)

    # crop the overlay from both images
    bh,bw = back.shape[:2]
    fh,fw = front.shape[:2]
    x1, x2 = max(x, 0), min(x+fw, bw)
    y1, y2 = max(y, 0), min(y+fh, bh)
    front_cropped = front[y1-y:y2-y, x1-x:x2-x]
    back_cropped = back[y1:y2, x1:x2]

    alpha_front = front_cropped[:,:,3:4] / 255
    alpha_back = back_cropped[:,:,3:4] / 255
    
    # replace an area in result with overlay
    result = back.copy()
    print(f'af: {alpha_front.shape}\nab: {alpha_back.shape}\nfront_cropped: {front_cropped.shape}\nback_cropped: {back_cropped.shape}')
    result[y1:y2, x1:x2, :3] = alpha_front * front_cropped[:,:,:3] + alpha_back * back_cropped[:,:,:3]
    result[y1:y2, x1:x2, 3:4] = (alpha_front + alpha_back) / (1 + alpha_front*alpha_back) * 255

    return result

def crop_make_img(img,bbox):
    col, row, ch = img.shape
    #bbox = coord
    #crop_img = img[y:y+h, x:x+w]
    x = int(row * bbox[1] - (row * bbox[3]) / 2)
    y = int(col * bbox[2] - (col * bbox[4]) / 2)
    w = int(row * bbox[3])
    h = int(col * bbox[4])
    print(w,h)
    #pt2 = (int(row * bbox[1] + (row * bbox[3]) / 2), int(col * bbox[2] + (col * bbox[4]) / 2))
    crop_img = img[y:y+h, x:x+w]

    scale_percent = 100 # percent of original size

    max_v=0
    if w > h:
      max_v = w
    else:
      max_v = h

    if max_v > 299:
      scale_percent = (299/max_v) * 100

    width = int(crop_img.shape[1] * scale_percent / 100)
    height = int(crop_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    print(dim)
    # resize image
    resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)

    new_x = int(150 - width/2)
    new_y = int(150 - height/2)
    im = np.zeros((299,299,3),dtype='uint8')
    
    final = merge_image(im,resized,new_x, new_y)
    return final

for c in range(0, num_of_class):
  d = ""+str(c)
  path = os.path.join(dst_folder, d)
  os.mkdir(path)


#label_list = os.listdir(label_path2)

#실제 이미지 경로들을 저장한 파일을 읽어들이는 코드

lines = open("\\Users\\user\\Desktop\\modeling\\daejang_data\\full\\label_txt").readlines()

for f in lines:
            print(f)
            f = f[:-1]
            ff = os.path.basename(f)
            coord = list()
            
            image_path3 = ff[:-4]+".jpg"  
            
            label_path3 = "\\Users\\user\\Desktop\\modeling\\daejang_data\\full\\"+ff[:-4]+".txt"
            
            img = cv2.imread(image_path3)
            
            lines = open(label_path3).readlines()
            
            counter = 0
            for line in lines:
                    counter = counter + 1
                    line = line.split()
                    for i in range(0, len(line)):
                        coord.append(float(line[i]))
                    
                    final = crop_make_img(img,coord)
                    #img = draw_bbox(img, coord, type=type)
                    image_dd = dst_folder2+str(int(coord[0]))+"/" + ff[:-4]+str(counter)+".jpg"
                    print(image_dd)
                    cv2.imwrite(image_dd, final)
                    #img = draw_center(img, coord, type=type)
                    coord = list()
