# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:18:57 2019

@author: herb
"""

import os
from random import shuffle
import pandas as pd
#import loader_captcha as loader
#import loader_Oracle as loader
import loader_yizu as loader

def split(full_list,bshuffle=False,ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if bshuffle:
        shuffle(full_list)
        sublist_1 = full_list[:offset]
        sublist_2 = full_list[offset:]
        return sublist_1,sublist_2

'''
读取文件夹中的所有图片名称，将其完整路径存放在list中
'''
def file_read(file_dir):
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for filename in files:  
            if (os.path.splitext(filename)[1] == '.png' or os.path.splitext(filename)[1] == '.jpg') \
                        and filename[0] != '.': 
                L.append(os.path.join(root, filename))
    return L

def txt_write(filename, fdata):
    """把数据写入文本文件中"""
    file = open(filename, 'w')
    for i, data in enumerate(fdata):
        if i != len(fdata) - 1:
            file.write(str(data) + ',')
        else:
            file.write(str(data)) # 最后一个不加逗号
    file.close()

def read_img_all(files):
    # 一张一张png读取
    tables = []
    for ff in files:
        img_name = ff.split('\\')[7] # 第5的一个代表文字所属类别，跟所在目录有关
        tables.append(img_name)
    return tables

def save_table(files):
    #获取训练数据集
    tables = loader.set_table(files)
    # 将tables存放到本地
    tables = list(set(tables)) # 去除list中的相同元素
    print("数据集标签分类",len(tables))
    df = {'tables':tables[:]}
    df = pd.DataFrame(df)
    df.to_csv(loader.TABLES_DIR, index = False, mode = 'w', encoding='utf-8-sig') # 将标签的种类进行保存

if __name__=="__main__":
    # 读取所有文件的地址
    L = file_read(loader.FILE_DIR)
    print("数据集总数",len(L))
    # 将文件分为3部分，一部分测试，一部分交叉验证，一部分训练    
    L1,L2 = split(L,True,0.8)
    txt_write(loader.TEST_DIR,L2)
    print("测试集",len(L2))
    L1,L2 = split(L1,True,0.9)
    txt_write(loader.TRAIN_DIR,L1)
    print("训练集",len(L1))
    txt_write(loader.VALIDA_DIR,L2)
    print("交叉验证集",len(L2))
    
    # 生成对应label的table表
    save_table(L)
    
    
  