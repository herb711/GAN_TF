
'''
初始化模型
训练模型
'''

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.rcParams['backend']='SVG'
import matplotlib.pyplot as plt
import pickle

import network as net

import sys
sys.path.append("..")
from libs.unit import mkdir,clear_OOM
#import libs.loader_mnist as loader
import libs.loader_yizu as loader

    
# 解决内存泄露问题
clear_OOM()
# 新建目录
mkdir(loader)
    

# 定义模型参数
learning_rate = 0.001 # 学习率
beta1 = 0.4

# 定义输入输出参数
noise_size = 100
data_shape = [-1, loader.input_shape[0], loader.input_shape[1], loader.input_shape[2]]

# 定义训练参数
epochs = 10000
batch_size = 25
n_samples = 1000


def plot_images(n, samples):
    fig, axes = plt.subplots(nrows=1, ncols=25, sharex=True, sharey=True, figsize=(50,2))
    for img, ax in zip(samples, axes):
        ax.imshow(img.reshape((data_shape[1], data_shape[2])), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0)

    plt.savefig(loader.GAN_IMAGE_DIR + '/' + str(n) + '.svg',format='svg')
    #plt.savefig(loader.GAN_IMAGE_DIR + '/' + str(n) + '.png')
    #plt.show()
    #plt.pause(6)# 间隔的秒数：6s
    plt.close()

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


def train(data_train, data_test):
    """
    @Author: zhushiyu
    --------------------
    @param noise_size: 噪声size
    @param data_shape: 真实图像shape
    @epochs:           训练次数
    @batch_size:       小批量大小
    @n_samples:        间隔显示的频率
    """

    # 存储loss
    losses = []
    steps = 0
    mask_np1 = unit.mask_square(data_shape[1], data_shape[2])# 生成一副与图象同样大小的屏蔽图 
    
    inputs_real, inputs_noise = net.get_inputs(noise_size, data_shape[1], data_shape[2], data_shape[3])
    g_loss, d_loss = net.get_loss(inputs_real, inputs_noise, data_shape[-1])
    g_train_opt, d_train_opt = net.get_optimizer(g_loss, d_loss, beta1, learning_rate)

    z_train_opt, z_img, z_loss = net.get_opt_z(noise_size, data_shape, inputs_real, mask_np1, learning_rate)
      
    # 保存生成器变量
    saver = tf.train.Saver()
    
    # 读取png版本
    #test_img, _ = next(data_train)
    #test_img = test_img.reshape((batch_size, data_shape[1], data_shape[2], data_shape[3]))
    #plot_images(0,np.squeeze(test_img, -1)) # 显示完整图像

    # 生成待修复图像
    #mask_np_ = mask_np.reshape(data_shape[1], data_shape[2], data_shape[3])
    #batch_mix = [np.multiply(mask_np_, y) for y in test_img] # 点乘
    #plot_images(1,np.squeeze(batch_mix, -1)) # 显示残缺图像    


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # 变量初始化
        
        # 载入模型
        ckpt = tf.train.get_checkpoint_state(loader.MODEL_DIR) # 通过checkpoint文件自动找到目录中最新模型的文件名
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path) # 加载模型   
        
        # 迭代epoch
        for e in range(epochs):
            steps += 1
            
            # 读取png版本
            batch_images, _ = next(data_train)
            batch_images = batch_images.reshape((batch_size, data_shape[1], data_shape[2], data_shape[3]))

            # noise
            batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))

            # run optimizer
            _ = sess.run(g_train_opt, feed_dict={inputs_real: batch_images,
                                                 inputs_noise: batch_noise})
            _ = sess.run(d_train_opt, feed_dict={inputs_real: batch_images,
                                                 inputs_noise: batch_noise})
            
            if steps % n_samples == 0:
                train_loss_g = g_loss.eval({inputs_real: batch_images,
                                            inputs_noise: batch_noise})
                train_loss_d = d_loss.eval({inputs_real: batch_images,
                                            inputs_noise: batch_noise})
                
                test_images, _ = next(data_test)
                test_loss_d = d_loss.eval({inputs_real: test_images,
                                            inputs_noise: batch_noise})

                # 寻找最佳的伪图像
                batch_mix = [batch_images[0]]

                for i in range(1):# 2000
                    _, train_img_z, train_loss_z = sess.run([z_train_opt, z_img, z_loss], 
                                               feed_dict={inputs_real: batch_mix})
                    
                losses.append((train_loss_d, test_loss_d, train_loss_g, train_loss_z))
                
                # 显示图片
                samples = show_generator_output(sess, batch_noise, inputs_noise, data_shape[-1])
                plot_images(steps,samples)

                print("Epoch {}/{}....".format(e+1, epochs), 
                      "D_Loss: {:.4f}....".format(train_loss_d),
                      "G_Loss: {:.4f}....". format(train_loss_g))
                      #"D_V_Loss: {:.4f}....". format(test_loss_d),
                      #"Z_Loss: {:.4f}....". format(np.mean(train_loss_z)))
                
                # 每10次保存1次模型
                saver.save(sess, loader.MODEL_DIR + r'/generator'+ str(steps) + '.ckpt')

        # 训练结束保存模型
        saver.save(sess, loader.MODEL_DIR + r'/generator.ckpt')

    return losses

if __name__=="__main__":

    # 读取数据
    data_test = loader.read_batch(loader.TEST_DIR,batch_size)
    batchs = loader.read_batch(loader.TRAIN_DIR,batch_size)
    
    lossdir = os.path.join(loader.HISTORY_DIR, 'DCGAN-gan') + '.loss.pl'
    
    # 训练
    with tf.Graph().as_default():
        losses = train(batchs, data_test)
        with open(lossdir, 'wb') as f:
            pickle.dump(losses, f)
            f.close()
    
    # 显示
    with open(lossdir, 'rb') as f:
        losses = pickle.load(f)
        f.close()
        
        fig, ax = plt.subplots(figsize=(20,7))
        losses = np.array(losses)
        #plt.plot(losses.T[0]+losses.T[1], label='Discriminator Total Loss')
        plt.plot(losses.T[0], label='Discriminator Loss')
        plt.plot(losses.T[1], label='Generator Loss')
        #plt.plot(losses.T[2], label='Z Loss')
        plt.title("Training Losses")
        plt.legend()
        #plt.savefig(loader.HISTORY_DIR + r'/tarin_losses.png')
        imgdir = os.path.join(loader.HISTORY_DIR, 'DCGAN-gan') + '.loss.svg'
        plt.savefig(imgdir,format='svg')
        plt.show()
        

    
    

