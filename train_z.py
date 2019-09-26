
'''
初始化模型
训练模型
'''

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

import loader_yizu as loader
import network as net



# 定义模型参数
learning_rate = 0.001 # 学习率
beta1 = 0.4

# 定义输入输出参数
noise_size = 100
data_shape = [-1, 64, 64, 1]

# 定义训练参数
epochs = 5
epochs_z = 2000
batch_size = 1 # 只能设置成1 因为每一副图对应的都不一样
n_samples = 1

# 生成屏蔽区域
m = 20
n = 40
mask_np = np.ones([data_shape[1],data_shape[2]]) #生成一个64x64的全1矩阵
mask_np[m:n, m:n]= 0

mask_np0 = np.zeros([data_shape[1],data_shape[2]]) #生成一个64x64的全0矩阵
mask_np0[m:n, m:n]= 1

    
def plot_images3(n, samples):
    fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(25,8))
    for img, ax in zip(samples, axes):
        ax.imshow(img.reshape((64, 64)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0)
    
    # 保存图片
    plotPath = os.getcwd() + "\\checkpoints\\fiximgs\\"
    if not os.path.exists(plotPath): # 如果目录不存在，新建
        os.mkdir(plotPath)

    plt.savefig(plotPath + str(n) + '.png')
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



def train_z(data_test, noise_size, data_shape, n_samples):
    """
    @Author: zhushiyu
    --------------------
    @param noise_size: 噪声size
    @param data_shape: 真实图像shape
    @n_samples: 显示示例图片数量
    """
    
    inputs_real, inputs_noise = net.get_inputs(noise_size, data_shape[1], data_shape[2], data_shape[3])
    g_loss, d_loss = net.get_loss(inputs_real, inputs_noise, data_shape[-1])
    g_train_opt, d_train_opt = net.get_optimizer(g_loss, d_loss, beta1, learning_rate)

    z_train_opt, z_img, z_loss = net.get_opt_z(noise_size, data_shape, inputs_real, mask_np, learning_rate)
      
    # 保存生成器变量
    saver = tf.train.Saver()
    

    with tf.Session() as sess: # 在默认的图上创建会话
        modelone_path = './checkpoints/'
        ckpt = tf.train.get_checkpoint_state(modelone_path) # 通过checkpoint文件自动找到目录中最新模型的文件名
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path) # 加载模型    
            for n in range(epochs):
                # 读取png图片---从测试集中读取
                test_img, _ = next(data_test)
                test_img = test_img.reshape(data_shape[1], data_shape[2])
                #plot_images(np.squeeze(test_img, -1)) # 显示完整图像
    
                # 生成待修复图像
                batch_mix = [np.multiply(mask_np, test_img) for x in range(batch_size)] # 点乘
                batch_mix = np.array(batch_mix)
                batch_mix = batch_mix.reshape((batch_size, data_shape[1], data_shape[2], data_shape[3]))
                #plot_images(np.squeeze(batch_mix, -1)) # 显示残缺图像    
    
                losses = []
                # 寻找最佳的伪图像
                for i in range(epochs_z):
                    _, train_img_z, train_loss_z = sess.run([z_train_opt, z_img, z_loss], 
                                               feed_dict={inputs_real: batch_mix})
                    losses.append(train_loss_z)
    
                # 生成最佳的伪图像
                zz_img = show_generator_output(sess, train_img_z, inputs_noise, data_shape[-1])
                #plot_images(z_img)  # 显示伪图像
                zz_img = zz_img.reshape(data_shape[1], data_shape[2])
    
                # 进行修复
                fix_img = np.multiply(mask_np0, zz_img)  + batch_mix[0].reshape(data_shape[1], data_shape[2])
                #plot_images(fix_img.reshape((1, 64, 64))) # 显示修补之后的图像
    
                # 显示图像
                samples = []
                samples.append(test_img)
                samples.append(batch_mix[0])
                samples.append(zz_img)
                samples.append(fix_img)
                plot_images3(n, samples)
                
                plt.plot(losses)
                plt.show()


if __name__=="__main__":

    # 读取数据
    #data_train, data_test = loader.read_yizu_batch(batch_size)
    batchs = loader.read_batch(loader.TEST_DIR, batch_size)
    
    # 训练
    with tf.Graph().as_default():
        train_z(batchs, noise_size, [-1, 64, 64, 1], n_samples)
    
    
    
