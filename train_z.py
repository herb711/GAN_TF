
'''
初始化模型
训练模型
'''

import os
import tensorflow as tf
import numpy as np
import pickle
import matplotlib
matplotlib.rcParams['backend']='SVG'
import matplotlib.pyplot as plt

import network as net

import sys
sys.path.append("..")
import libs.unit as unit
from libs.unit import mkdir,clear_OOM
#import libs.loader_mnist as loader
import libs.loader_yizu as loader

# 解决内存泄露问题
clear_OOM()
# 新建目录
mkdir(loader)

# 设置loss保存地址
lossdir = os.path.join(loader.HISTORY_DIR, 'DCGAN-d') + '.loss.pl'
losssvgdir = os.path.join(loader.HISTORY_DIR, 'DCGAN-d') + '.loss.svg'


# 定义模型参数
learning_rate = 0.001 # 学习率
beta1 = 0.4

# 定义输入输出参数
noise_size = 100
data_shape = [-1, loader.input_shape[0], loader.input_shape[1], loader.input_shape[2]]

# 定义训练参数
epochs = 5
epochs_z = 2000
batch_size = 1 # 只能设置成1 因为每一副图对应的都不一样
n_samples = 1

def plot_image(n, img):
    plt.imshow(img.reshape((data_shape[1], data_shape[2])), cmap='Greys_r')
    plt.savefig(loader.FIX_IMAGE_DIR + '/' + str(n) + '.png')
    plt.close()
    
def plot_images3(n, samples):
    fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(25,8))
    for img, ax in zip(samples, axes):
        ax.imshow(img.reshape((data_shape[1], data_shape[2])), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0)
    
    # 保存图片
    plt.savefig(loader.FIX_IMAGE_DIR + '/' + str(n) + '.svg',format='svg')
    #plt.savefig(loader.FIX_IMAGE_DIR + '/' + str(n) + '.png')
    #plt.show()
    #plt.pause(6)# 间隔的秒数：6s
    plt.close()

'''
对每幅图像的loss进行批量显示
'''
def plot_images_losses(losses,num=5):#num为要显示的数据项数默认为5
    fig = plt.gcf()
    fig.set_size_inches(25, 20)
    losses = np.array(losses)
    if num>5: num=5 
    for i in range(0, num):

        ax=plt.subplot(5,1, 1+i)#子图形为5行5列
        
        ax.plot(losses.T[i], label='Loss')
        
        title= str(i) + " Training Losses" 
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])
    
    plt.savefig(losssvgdir,format='svg')
    plt.show()    

def show_generator_output(sess, examples_noise, inputs_noise, output_dim):
    """
    @Author: zhushiyu
    --------------------
    @param sess: TensorFlow session
    @param n_images: 展示图片的数量
    @param inputs_noise: 噪声图片
    @param output_dim: 图片的depth（或者叫channel）
    @param image_mode: 图像模式：RGB或者灰度
    """

    # 生成噪声图片
    samples = sess.run(net.get_generator(inputs_noise, output_dim, False),
                       feed_dict={inputs_noise: examples_noise})

    result = np.squeeze(samples, -1)
    return result


def save_generator_output(img, mdir, name):
    """
    @Author: zhushiyu
    将是生成的图像存储在本地
    --------------------
    @param img: 存储的图像
    @param mdir: 存储的地址
    """
    
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(data_shape[1],data_shape[2]))
    axes.imshow(img.reshape((data_shape[1],data_shape[2])), cmap='Greys_r')
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0)
        
    if not os.path.exists(mdir):
        os.makedirs(mdir)
    plt.savefig(mdir + '/' + str(name) + '.png')
    
    plt.close()



def train_z(data_test, noise_size, data_shape, n_samples):
    """
    @Author: zhushiyu
    --------------------
    @param noise_size: 噪声size
    @param data_shape: 真实图像shape
    @n_samples: 显示示例图片数量
    """
    
    mask_np1 = unit.mask_square(data_shape[1], data_shape[2])# 生成一副与图象同样大小的屏蔽图 
    
    inputs_real, inputs_noise = net.get_inputs(noise_size, data_shape[1], data_shape[2], data_shape[3])
    g_loss, d_loss = net.get_loss(inputs_real, inputs_noise, data_shape[-1])
    g_train_opt, d_train_opt = net.get_optimizer(g_loss, d_loss, beta1, learning_rate)

    z_train_opt, z_img, z_loss = net.get_opt_z(noise_size, data_shape, inputs_real, mask_np1, learning_rate)
      
    # 保存生成器变量
    saver = tf.train.Saver()
    
    losses = []

    with tf.Session() as sess: # 在默认的图上创建会话
        ckpt = tf.train.get_checkpoint_state(loader.MODEL_DIR) # 通过checkpoint文件自动找到目录中最新模型的文件名
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path) # 加载模型    
            # 读取png图片---从测试集中读取
            test_img, test_label = next(data_test)
            test_img = test_img.reshape(data_shape[1], data_shape[2])
            #plot_images(np.squeeze(test_img, -1)) # 显示完整图像
            
            #mask_np1 = unit.mask_img(data_shape[1], data_shape[2])# 生成一副与图象同样大小的屏蔽图 
            batch_mix = [np.multiply(mask_np1, test_img) for x in range(batch_size)] # 点乘
            batch_mix = np.array(batch_mix)
            batch_mix = batch_mix.reshape((batch_size, data_shape[1], data_shape[2], data_shape[3]))
            #plot_images(np.squeeze(batch_mix, -1)) # 显示残缺图像    

            # 寻找最佳的伪图像
            for i in range(epochs_z):
                _, train_img_z, train_loss_z = sess.run([z_train_opt, z_img, z_loss], 
                                           feed_dict={inputs_real: batch_mix})
                #plot_image(i,train_img_z)  # 显示伪图像
                losses.append(train_loss_z)

            # 生成最佳的伪图像
            zz_img = show_generator_output(sess, train_img_z, inputs_noise, data_shape[-1])
            #plot_image(i,zz_img)  # 显示伪图像
            zz_img = zz_img.reshape(data_shape[1], data_shape[2])

            # 进行修复
            mask_np0 = 1-mask_np1
            fix_img = np.multiply(mask_np0, zz_img)  + batch_mix[0].reshape(data_shape[1], data_shape[2])
            #plot_images(fix_img.reshape((1, data_shape[1], data_shape[2]))) # 显示修补之后的图像

            # 显示图像
            samples = []
            samples.append(test_img)     # 原始图像
            samples.append(batch_mix[0]) # 残缺图像
            samples.append(zz_img)       # 生成图像
            samples.append(fix_img)      # 修复图像
            plot_images3(n_samples, samples)
            
            #num = loader.predic_label_h(test_label) # 求出数字实际值

            #plt.imshow(test_img, cmap='Greys_r')
            #plt.savefig(loader.FIX_IMAGE_DIR1 + '/' + str(n) + '.png')
            
            #mdir = loader.FIX_IMAGE_DIR2 + '/' + str(num)
            #save_generator_output(batch_mix[0], mdir, n)
            
            #plt.imshow(zz_img, cmap='Greys_r')
            #plt.savefig(loader.FIX_IMAGE_DIR3 + '/' + str(n) + '.png')

            #mdir = loader.FIX_IMAGE_DIR4 + '/' + str(num)
            #save_generator_output(fix_img, mdir, n)

                
    return losses

if __name__=="__main__":

    # 读取数据
    #data_train, data_test = loader.read_yizu_batch(batch_size)
    batchs = loader.read_batch(loader.TEST_DIR, batch_size)
        
    # 预测
    #for n in range(epochs):
    with tf.Graph().as_default():
        losses = train_z(batchs, noise_size, [-1, data_shape[1], data_shape[2],data_shape[3]], 0)
        with open(lossdir, 'wb') as f:
            pickle.dump(losses, f)
            f.close()
    
    # 显示
    with open(lossdir, 'rb') as f:
        losses = pickle.load(f)
        f.close()
        #plot_images_losses(losses,5)
        plt.plot(losses)
        plt.savefig(losssvgdir,format='svg')
        plt.show()
        plt.close()

 
    

