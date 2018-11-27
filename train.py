
'''
初始化模型
训练模型
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

import loader
import network as net



# 定义模型参数
learning_rate = 0.001
beta1 = 0.4

# 定义输入输出参数
noise_size = 100
data_shape = [-1, 64, 64, 1]

# 定义训练参数
epochs = 1
batch_size = 25
n_samples = 101

# 生成屏蔽区域
m = 20
n = 40
mask_np = np.ones([data_shape[1],data_shape[2]]) #生成一个64x64的全1矩阵
mask_np[m:n, m:n]= 0

mask_np0 = np.zeros([data_shape[1],data_shape[2]]) #生成一个64x64的全0矩阵
mask_np0[m:n, m:n]= 1




def plot_images(samples):
    fig, axes = plt.subplots(nrows=1, ncols=25, sharex=True, sharey=True, figsize=(50,2))
    for img, ax in zip(samples, axes):
        ax.imshow(img.reshape((64, 64)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0)
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


def train(data_train):
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
    
    inputs_real, inputs_noise = net.get_inputs(noise_size, data_shape[1], data_shape[2], data_shape[3])
    g_loss, d_loss = net.get_loss(inputs_real, inputs_noise, data_shape[-1])
    g_train_opt, d_train_opt = net.get_optimizer(g_loss, d_loss, beta1, learning_rate)

    z_train_opt, z_img, z_loss = net.get_opt_z(noise_size, data_shape, inputs_real, mask_np, learning_rate)
      
    # 保存生成器变量
    saver = tf.train.Saver()
    
    
    # 读取png版本
    test_img = random.sample(data_train, batch_size)
    test_img = np.array(test_img)
    test_img = test_img.reshape((batch_size, data_shape[1], data_shape[2], data_shape[3]))
    plot_images(np.squeeze(test_img, -1)) # 显示完整图像

    # 生成待修复图像
    mask_np_ = mask_np.reshape(data_shape[1], data_shape[2], data_shape[3])
    batch_mix = [np.multiply(mask_np_, y) for y in test_img] # 点乘
    plot_images(np.squeeze(batch_mix, -1)) # 显示残缺图像    


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 迭代epoch
        for e in range(epochs):
            for batch_i in range(30000//batch_size): ##(mnist.train.num_examples//batch_size):
                steps += 1
                
                # minst版本
                #batch = mnist.train.next_batch(batch_size)
                #batch_images = batch[0].reshape((batch_size, data_shape[1], data_shape[2], data_shape[3]))
                
                # 读取csv版本
                #batch = data_train.sample(batch_size)
                #batch = batch.as_matrix()   
                
                # 读取png版本
                batch = random.sample(data_train, batch_size)
                batch = np.array(batch)               
                batch_images = batch.reshape((batch_size, data_shape[1], data_shape[2], data_shape[3]))
                
                # scale to -1, 1
                #batch_images = batch_images * 2 - 1

                # noise
                batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))

                # run optimizer
                _ = sess.run(g_train_opt, feed_dict={inputs_real: batch_images,
                                                     inputs_noise: batch_noise})
                _ = sess.run(d_train_opt, feed_dict={inputs_real: batch_images,
                                                     inputs_noise: batch_noise})
                
                if steps % n_samples == 0:
                    train_loss_d = d_loss.eval({inputs_real: batch_images,
                                                inputs_noise: batch_noise})
                    train_loss_g = g_loss.eval({inputs_real: batch_images,
                                                inputs_noise: batch_noise})
                    
                    _, train_img_z, train_loss_z = sess.run([z_train_opt, z_img, z_loss], 
                                                   feed_dict={inputs_real: batch_mix})
                    
                    losses.append((train_loss_d, train_loss_g))
                    # 显示图片
                    samples = show_generator_output(sess, batch_noise, inputs_noise, data_shape[-1])
                    plot_images(samples)

                    samples = show_generator_output(sess, train_img_z, inputs_noise, data_shape[-1])
                    plot_images(samples)

                    
                    print("Epoch {}/{}....".format(e+1, epochs), 
                          "Discriminator Loss: {:.4f}....".format(train_loss_d),
                          "Generator Loss: {:.4f}....". format(train_loss_g),
                          "Z Loss: {:.4f}....". format(np.mean(train_loss_z)))
                    saver.save(sess, './checkpoints/generator.ckpt')
    return losses

if __name__=="__main__":
    data_train, data_test = loader.read_yizu()
    with tf.Graph().as_default():
        losses = train(data_train)
        with open('train_losses.pkl', 'wb') as f:
            pickle.dump(losses, f)
            f.close()


    
    

