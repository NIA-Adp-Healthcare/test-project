"""
yolo 모델을 학습시키기위해 필요한 annotation txt 파일을 생성하는 script다.
"""

import json
import os, glob
import time
from tqdm import tqdm

def preprocessing_json(json_dir, save_path):
    json_list = get_json_list(json_dir)

    for _,json_name in enumerate(tqdm(json_list)):
        data = open_json(json_dir+"\\"+json_name)
        anot_list = get_annotation(data)
        txt_name = json_name.split('\\')[-1][:-4] + 'txt'
        if anot_list:
            save_txt(save_path, txt_name, anot_list)
        time.sleep(0.005)

"""
yolo 모델의 경우 annotation의 형식으로 center(x,y) 좌표와 width, height의 상댓값을 필요로한다.
"""
def get_yolo(point_list):
    x_min, y_min, x_max, y_max = get_points(point_list)

    center_x = (x_min + x_max)/(2*640)
    center_y = (y_min + y_max)/(2*480)
    width = (x_max - x_min)/640
    height = (y_max - y_min)/480
    point = str(center_x)+" "+str(center_y)+" "+str(width)+" "+str(height)

    return point

def get_points(point_list):
    x_list = [point_list[i][0] for i in range(len(point_list))]
    y_list = [point_list[j][1] for j in range(len(point_list))]

    x_min = min(x_list)
    y_min = min(y_list)
    x_max = max(x_list)
    y_max = max(y_list)

    return x_min, y_min, x_max, y_max

"""
json파일에서 phase_id의 값이 환자의 병기를 나타낸다.
"""
def get_stage(json_data):
    return str(json_data['image']['phase_id']-1)

def open_json(data_path):
    with open(data_path, 'r') as f:
        json_data = json.load(f)
    f.close()

    return json_data

def get_json_list(json_dir):
    json_list=list()

    print("Searching for json files...")
    os.chdir(json_dir)
    f_list = glob.glob("**/*.json", recursive=True)
    for f in f_list:
        if f.split('\\')[-2]=='ENDO':
            json_list.appned(f)
    print("Start creating text files...")

    return json_list

def get_annotation(json_data):
    anot_list = list()

    for anot in json_data['image']['annotations']:
        polygon = anot['polygon']
        point = get_stage(json_data) + " " + get_yolo(polygon)
        anot_list.append(point)

    return anot_list

def save_txt(save_path, filename, anot_list):
    with open(save_path + "\\" + filename, 'w') as f:
        f.write('\n'.join(anot_list))
    f.close()

if __name__ == '__main__':
    json_dir = r"C:\Users\whdud\Documents\GitHub\deeplearningMedi2020\DataForSubmit\forSubmit\NIA_JSON_C16"
    save_path = r"C:\Users\whdud\Desktop\temp\C16_txt"

    preprocessing_json(json_dir, save_path)