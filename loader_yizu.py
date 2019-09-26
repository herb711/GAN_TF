# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:48:52 2019

对彝族文字进行加载

@author: herb
"""
import os
import pandas as pd
import numpy as np
from random import shuffle
import cv2

# 图像存放目录
UP_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../..")) #获取绝对路径 + 返回上上层
FILE_DIR = UP_DIR + r'/datasets/yizu/images_labels/'
TRAIN_DIR = FILE_DIR + r'/train.txt'
VALIDA_DIR = FILE_DIR + r'/valida.txt'
TEST_DIR = FILE_DIR + r'/test.txt'
TABLES_DIR = FILE_DIR + r'/tables.csv' # 标签分类存放目录

# 模型存放目录
GAN_IMAGE_DIR = r'images/yizu_%d.png'

MODEL_DIR = r'./result/yizu/SaveModel'
HISTORY_DIR = 'loss.pl' # 设置loss保存地址

input_shape = (64,64,1)
output_shape = 1764

def change_label(y):
    # 将数字转换为数组
    e = np.zeros((output_shape))
    e[int(y)] = 1.0
    return e 

'''
读取单幅图片
并进行归一化
'''
def get_img(file_name):
    #img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE) # 直接读取单通道
    img = cv2.imread(file_name)
    img = cv2.resize(img,(input_shape[0], input_shape[1]),interpolation=cv2.INTER_CUBIC) # 插值法放大图像
    if (input_shape[2] == 3): # 如果是3通道，转换为3通道
        img = cv2.cvtColor(cv2.resize(img, (input_shape[0], input_shape[1])), cv2.COLOR_GRAY2BGR) #cv2.COLOR_GRAY2BGR
        img = img.reshape((input_shape[0], input_shape[1], input_shape[2]))
    else:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) # 将图像转化为灰度图
        img = cv2.resize(img, (input_shape[0], input_shape[1]))
        img = img.reshape((input_shape[0], input_shape[1], input_shape[2]))
    # 对图像像素进行scale，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
    img = (np.float32(img)-127.0)/128.0 # scale to -1, 1  
    return img


'''
从txt文件中读取文件名，存放在list中
'''
def file_read_txt(file_dir):
    stockList = 0
    with open(file_dir,"r") as stockFile:
        stockList=stockFile.read()
    if stockList != 0:
        setcode = []
        for eachStock in stockList.split(','):
            
            setcode.append(eachStock)                
        return setcode
    

'''
根据文件名或文件所在位置制作标签
'''
def get_table(file_dir):
    # 判断标签的文字对应目录文件是否存在
    if os.path.exists(file_dir): # 如果存在该文件，读取该文件，存入tables用于将名称替换为序号        
        # 读取标签序号
        mydf = pd.read_csv(file_dir)
        tables = mydf["tables"].values
        tables = np.array(tables).tolist()
        return tables

'''
一张一张png读取,获取标签
'''
def set_table(files):
    tables = []
    for fname in files:
        fname = fname.replace('\\','/')
        name = fname.split('/')[7]
        tables.append(name)
    return tables

'''
根据文件名或文件所在位置制作标签
'''
def get_label(tables,fname):
    fname = fname.replace('\\','/')
    img_name = fname.split('/')[7] # 第5的一个代表文字所属类别，跟所在目录有关
    i = tables.index(img_name)
    return change_label(i)
    

def read_batch(data_dir, batch_size):
    # 从txt中获取图片目录
    files = file_read_txt(data_dir)
    print("数据集个数：",len(files))
    tables = get_table(TABLES_DIR)
    shuffle(files) # 让文件名队列随机化
    batchs_x = []
    batchs_y = []
    while(1):
        for ff in files:        
            img = get_img(ff)
            batchs_x.append(img)
            label = get_label(tables,ff)
            batchs_y.append(label)
            if len(batchs_x) >= batch_size:
                yield (np.array(batchs_x), np.array(batchs_y)) # 返回迭代器
                batchs_x = []
                batchs_y = []

import matplotlib.pyplot as plt
'''
将预测出的标签转换为对应的字符编码
'''
def predic_label_h(y):
    #y = y.reshape(-1)
    #d = np.argmax(y,1)
    y = y.tolist()    
    d = y.index(max(y)) # 返回最大值的序号
    return predic_label(d)

def predic_label(d):
    tables = get_table(TABLES_DIR)
    label = tables[d] #从序号里找到对应的字符编码
    return label

'''
对图像和标签批量显示
'''
def plot_images_labels_prediction(images,labels,
                                  prediction,idx,num=10):#num为要显示的数据项数默认为10
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)#子图形为5行5列
        
        ax.imshow(images[idx,:,:,0],cmap='gray')
        title= "label=" +str(predic_label_h(labels[idx]))
        if len(prediction)>0:
            title+=",predict="+str(predic_label(prediction[idx]))
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()

if __name__=="__main__":
    batchs = read_batch(TEST_DIR,32)
    x_test, y_test = next(batchs)
    
    # 从零开始查看，查看十个数据
    plot_images_labels_prediction(x_test, y_test, [], 0, 10)

    
    
    
    
    
    
    
    
    