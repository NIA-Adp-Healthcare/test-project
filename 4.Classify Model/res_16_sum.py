import os
import sys
import pandas as pd
#import os
#import sys

src = pd.read_csv("./res_16_final.csv",usecols= ['img_name','prediction', 'True Value'])
 
rid_list=[]

dict_rid = {}
t_value_dict = {}
for i,row in src.iterrows():
  
  img_name = row['img_name']
  rid =img_name.split("/")[-1].split("_")[0] 
  rid =rid.lstrip("0")
  t_value_dict[rid] = int(row['True Value']) 
  p_value = row['prediction']
  key_v = str(rid)+"_"+str(p_value)
  if key_v in dict_rid:
    dict_rid[key_v] += 1 
  else:
    dict_rid[key_v] = 1

  if rid in rid_list:
     print("hi")
  else:
     rid_list.append(str(rid))


true_count = 0
false_count = 0

for rid in rid_list:

  max_label = 0 
  max_count = -1 
  key_v = str(rid)+"_0"
  if key_v in dict_rid:
    if dict_rid[key_v] > max_count:
        max_label = 0
        max_count = dict_rid[key_v] 
  key_v = str(rid)+"_1"
  if key_v in dict_rid:
    if dict_rid[key_v] > max_count:
        max_label = 1
        max_count =dict_rid[key_v]
  key_v = str(rid)+"_2"
  if key_v in dict_rid:
    if dict_rid[key_v] > max_count:
        max_label = 2
        max_count = dict_rid[key_v]
  key_v = str(rid)+"_3"
  if key_v in dict_rid:
    if dict_rid[key_v] > max_count:
        max_label = 3
        max_count = dict_rid[key_v]
  key_v = str(rid)+"_4"
  if key_v in dict_rid:
    if dict_rid[key_v] > max_count:
        max_label = 4
        max_count = dict_rid[key_v]


  if t_value_dict[rid] == max_label:
     true_count += 1
  else:
     false_count += 1 

print("true count >", true_count)
print("false count >", false_count)

print("accuracy > ", true_count/(true_count+false_count))
