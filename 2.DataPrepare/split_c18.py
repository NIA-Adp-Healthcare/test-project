
import sys
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split


# split_dataset(img_list, ALL_TEST, ALL_VALID, ALL_TRAIN)


def split_dataset(file_list, test_rid_list, valid_rid_list, train_rid_list):
    print("FILES", len(file_list))
    print("TEST LIST", len(test_rid_list))
    print("VALID LIST", len(valid_rid_list))
    print("TRAIN LIST", len(train_rid_list))

    all = pd.DataFrame({"path": file_list}) #series 생성

    all["rid"] = all["path"].apply(lambda x: x.split('/')[-1].split('_')[1].lstrip('txt')[1:].rstrip('.txt'))
    print(all["rid"][:10])
    all["rid"] = all["rid"].apply(lambda x: x.lstrip("0"))
    # all["rid"] = all['path']
    # print(all)
    print(all["rid"])
    all["isTest"] = all["rid"].apply(lambda x: x in test_rid_list)
    all["isValid"] = all["rid"].apply(lambda x: x in valid_rid_list)
    all["isTrain"] = all["rid"].apply(lambda x: x in train_rid_list)
    print(all["isTest"])
    forTest = all[all["isTest"] == True]
    forValid = all[all["isValid"] == True]
    forTrain = all[all["isTrain"] == True]
    #x_train , x_valid = train_test_split(forTrain, test_size=(1-train_ratio), random_state=1234)
    A = list(forTrain["path"])
    B = list(forValid["path"])
    C = list(forTest["path"])
    return A, B, C

####################

df = pd.read_csv('\\Users\\user\\Desktop\\modeling\\daejang_data\\full\\test_list.csv', dtype={'RID': str})
df_test = df[df["GUBUN"] == "TEST"]
ALL_TEST = list(df_test["rid"].unique())
ALL_TEST.sort()
print(ALL_TEST)
print()

df2 = pd.read_csv('\\Users\\user\\Desktop\\modeling\\daejang_data\\full\\valid_list.csv', dtype={'RID': str})
df_valid = df2[df2["GUBUN"] == "VALID"]
df_train = df2[df2["GUBUN"] == "TRAIN"]
ALL_VALID = list(df_valid["rid"].unique())
ALL_VALID.sort()
print(ALL_VALID)
print()

ALL_TRAIN = list(df_train["rid"].unique())
ALL_TRAIN.sort()
print(ALL_TRAIN)
print()

img_list = glob.glob('/Users/user/Desktop/modeling/daejang_data/full/label_txt/*.txt')
img_list.sort()
print(len(img_list))
print(img_list[:10])

train_img_list, val_img_list, test_list = split_dataset(img_list, ALL_TEST, ALL_VALID, ALL_TRAIN)

print(len(train_img_list), len(val_img_list), len(test_list))

with open('\\Users\\user\\Desktop\\modeling\\daejang_data\\full\\anewtrainByRid18.txt', 'w') as f:
  f.write('\n'.join(train_img_list) + '\n')

with open('\\Users\\user\\Desktop\\modeling\\daejang_data\\full\\anewvalByRid18.txt', 'w') as f:
  f.write('\n'.join(val_img_list) + '\n')

with open('\\Users\\user\\Desktop\\modeling\\daejang_data\\full\\anewtestByRid18.txt', 'w') as f:
  f.write('\n'.join(test_list) + '\n')

"""
import yaml

with open('./data.yaml', 'r') as f:
  data = yaml.load(f)

print(data)

data['train'] = '/home/joyhyuk/python/y2/train.txt'
data['val'] = '/home/joyhyuk/python/y2/val.txt'

with open('./data.yaml', 'w') as f:
  yaml.dump(data, f)

print(data)
"""

