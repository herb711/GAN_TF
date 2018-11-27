
'''
从数据集中读取文字图片
'''

import gc
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import random
import os

'''
# 从打包好的csv里读取图片
chunks =[]
for i in range(20):
    PATH_SAVE = '../dataset/CASIA_HWDB/images64_' + str(i) + '.csv'
    chunk = pd.read_csv(PATH_SAVE)# 一次性读取
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
data_train = df
data_train = data_train * 2 - 1 # scale to -1, 1
batchs = data_train.sample(10)
np_batchs = batchs.as_matrix()
plt.imshow(np_batchs[0].reshape((64, 64)), cmap='Greys_r')
plt.show()
print(data_train.shape)

'''

'''
# 一张一张png读取
import random
chunks = []

for i in range(33200):
    ff = '/Users/herb/workspace/dataset/images64/' + str(i+1) + '.png'
    img = io.imread(ff)
    chunks.append(img[:,:,0])
# 对图像像素进行scale，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
chunks = np.float32(chunks)
data_train = (chunks)/255 # 归一化 0-1;  -1~1: x/128-1
data_train = data_train * 2 - 1 # scale to -1, 1
data_train = data_train.tolist()

batchs = random.sample(data_train, 10)
batchs = np.array(batchs)
plt.imshow(batchs[0].reshape((64, 64)), cmap='Greys_r')
print(batchs.shape)


'''

# 读取彝族文字
def file_name(file_dir):
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for filename in files:  
            if os.path.splitext(filename)[1] == '.png' and filename[0] != '.': 
                L.append(os.path.join(root, filename))  
    return L  

def getdata(data_dir):
    # 一张一张png读取
    files = file_name(data_dir)
    chunks = []
    for ff in files:
        img = io.imread(ff)
        chunks.append(img)
    # 对图像像素进行scale，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
    chunks = np.float32(chunks)
    data = (chunks)/65535 # 归一化 0-1;  -1~1: x/128-1
    data = data * 2 - 1 # scale to -1, 1
    
    for x in locals().keys():
      del locals()[x]
    gc.collect()
    return data.tolist()

def read_yizu():

    #获取训练数据集
    data_train = getdata('../../dataset/yizu/images64/train/')
    
    #获取测试数据集
    data_test = getdata('../../dataset/yizu/images64/test/')
    
    print(len(data_train),  len(data_test))

    for x in locals().keys():
      del locals()[x]
    gc.collect()
    return data_train, data_test


if __name__=="__main__":
    data_train, data_test = read_yizu()
    
    batchs = random.sample(data_train, 10)
    batchs = np.array(batchs)
    plt.imshow(batchs[0].reshape((64, 64)), cmap='Greys_r')
    print(batchs.shape)
    
    noise = np.ones([64,64]) #生成一个64x64的全1矩阵
    noise[20:36, 20:36]= 0
    bad = batchs[0]
    bad[20:36, 20:36]= 1
    plt.imshow(bad, cmap='Greys_r')


    
    

