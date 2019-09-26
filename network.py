'''
构建模型
'''

import tensorflow as tf

# # 构建模型
# 
# - inputs
# - generator
# - discriminator
# - loss
# - optimizer

# ## Inputs

# In[6]:


def get_inputs(noise_dim, image_height, image_width, image_depth):
    """
    @Author: zhushiyu
    --------------------
    :param noise_dim: 噪声图片的size
    :param image_height: 真实图像的height
    :param image_width: 真实图像的width
    :param image_depth: 真实图像的depth
    """ 
    inputs_real = tf.placeholder(tf.float32, [None, image_height, image_width, image_depth], name='inputs_real')
    inputs_noise = tf.placeholder(tf.float32, [None, noise_dim], name='inputs_noise')
    
    return inputs_real, inputs_noise


# # Generator

# In[7]:


def get_generator(noise_img, output_dim, is_train=True, alpha=0.01):
    """
    @Author: zhushiyu
    --------------------
    :param noise_img: 噪声信号，tensor类型
    :param output_dim: 生成图片的depth
    :param is_train: 是否为训练状态，该参数主要用于作为batch_normalization方法中的参数使用
    :param alpha: Leaky ReLU系数
    """
    
    with tf.variable_scope("generator", reuse=(not is_train)):
        # 100 x 1 to 8 x 8 x 512
        # 全连接层
        layer1 = tf.layers.dense(noise_img, 8*8*512)
        layer1 = tf.reshape(layer1, [-1, 8, 8, 512])
        # batch normalization
        layer1 = tf.layers.batch_normalization(layer1, training=is_train)
        # Leaky ReLU
        layer1 = tf.maximum(alpha * layer1, layer1)
        # dropout
        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)
        
        # 8 x 8 x 512 to 16 x 16 x 256
        layer2 = tf.layers.conv2d_transpose(layer1, 256, 3, strides=2, padding='same')
        layer2 = tf.layers.batch_normalization(layer2, training=is_train)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)
        
        # 16 x 16 x 256 to 32 x 32 x 128
        layer3 = tf.layers.conv2d_transpose(layer2, 128, 3, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=is_train)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)
        
        # 32 x 32 x 128 to 64 x 64 x 1
        logits = tf.layers.conv2d_transpose(layer3, output_dim, 3, strides=2, padding='same')
        
        outputs = tf.tanh(logits)

        return outputs


# ## Discriminator

# In[8]:


def get_discriminator(inputs_img, reuse=False, alpha=0.01):
    """
    @Author: zhushiyu
    --------------------
    @param inputs_img: 输入图片，tensor类型
    @param alpha: Leaky ReLU系数
    """
    
    with tf.variable_scope("discriminator", reuse=reuse):
        # 64 x 64 x 1 to 32 x 32 x 128
        # 第一层不加入BN
        layer1 = tf.layers.conv2d(inputs_img, filters=128, kernel_size=3, strides=2, padding='same')
        layer1 = tf.maximum(alpha * layer1, layer1) # Leaky ReLU
        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)
        
        # 32 x 32 x 128 to 16 x 16 x 256
        layer2 = tf.layers.conv2d(layer1, 256, 3, strides=2, padding='same')
        layer2 = tf.layers.batch_normalization(layer2, training=True)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)
        
        # 16 x 16 x 256 to 8 x 8 x 512
        layer3 = tf.layers.conv2d(layer2, 512, 3, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=True)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)
        
        # 8 x 8 x 512 to 8*8*512 x 1
        flatten = tf.reshape(layer3, (-1, 8*8*512))
        logits = tf.layers.dense(flatten, 1)
        outputs = tf.sigmoid(logits)
        
        return logits, outputs


# ## Loss

# In[9]:


def get_loss(inputs_real, inputs_noise, image_depth, smooth=0.05):
    """
    @Author: zhushiyu
    --------------------
    @param inputs_real: 输入图片，tensor类型
    @param inputs_noise: 噪声图片，tensor类型
    @param image_depth: 图片的depth（或者叫channel）
    @param smooth: label smoothing的参数
    """
    
    g_outputs = get_generator(inputs_noise, image_depth, is_train=True)
    d_logits_real, d_outputs_real = get_discriminator(inputs_real)
    d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, reuse=True)
     
    
    # 计算Loss
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                                    #labels=inputs_real*(1-smooth)))
                                                                    labels=tf.ones_like(d_outputs_fake)*(1-smooth))) # ones_like全部值置为1
    
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels=tf.ones_like(d_outputs_real)*(1-smooth)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.zeros_like(d_outputs_fake)))
    d_loss = tf.add(d_loss_real, d_loss_fake)
    
    return g_loss, d_loss


# ## Optimizer

# In[10]:


def get_optimizer(g_loss, d_loss, beta1=0.4, learning_rate=0.001):
    """
    利用梯度下降法对网络进行优化
    @Author: zhushiyu
    --------------------
    @param g_loss: Generator的Loss
    @param d_loss: Discriminator的Loss
    @learning_rate: 学习率
    """
    
    train_vars = tf.trainable_variables()
    
    g_vars = [var for var in train_vars if var.name.startswith("generator")]
    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
    
    # Optimizer
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
        d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    
    return g_opt, d_opt


# ## 增加的模型

# In[11]:


def get_opt_z(noise_size, data_shape, inputs_real, mask_np, learning_rate=0.001) :
    """
    @ Author: zhushiyu
    --------------------
    输入参数
    @ noise_size: 生成器的输入为随机噪声，这里设置噪声的长度
    @ data_shape: 生成器生成出来的图像的shape
    @ inputs_real:      输入的待修复的图像
    @ mask_np:          屏蔽的区域
    @ learning_rate:    学习率
    --------------------
    返回值
    z_opt:   运行开关，进行学习需要
    layer_z: 生成器的输入，每次根据此模型而进行改变
    complete_loss: 总的损失值
    --------------------    
    # 寻找与待修复图像最接近的伪图像 
    
    """
    
    # 生成器的输入
    z = tf.random_uniform([noise_size], -1.0, 1.0) # 随机噪声，用于初始化
    var_z = tf.Variable(initial_value=z, name='batch_z', dtype=tf.float32) # 建立变量, 初始值是随机噪声，之后随模型优化改变
    layer_z = tf.reshape(var_z, [1, noise_size]) # 数据类型进行变化。因为每次只输入一个数据，因此是[1, noise_size]

    # 生成屏蔽器
    mask_np_ = mask_np.reshape(1, data_shape[1], data_shape[2], data_shape[3])
    mask = tf.constant(value=mask_np_, dtype=tf.float32)

    # 双判别器模型结构，获取判别器1计算出的结果
    z_images = get_generator(layer_z, data_shape[-1], False) # in (1x100)  out (1x64x64x1)
    z_logits_fake, z_outputs_fake = get_discriminator(z_images, reuse=True) # in (1x64x64x1)  out (1)
    z_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z_logits_fake, 
                                                                    labels=tf.ones_like(z_outputs_fake)*(1-0.1))) # 计算Loss
    perceptual_loss = z_loss # 判别器认定为真的情况
    
    # 计算生成图像和原始图像差异的loss值，即伪图像跟真实图像接近情况
    b_img = tf.reshape(inputs_real[0], [1, data_shape[1], data_shape[2], 1]) 
    contextual0 = tf.multiply(mask, z_images) - tf.multiply(mask, b_img) # 首先和非屏蔽区域相乘，把需要修补的部分排除在外；然后两幅图相减获取差距
    contextual1 =  tf.contrib.layers.flatten( tf.abs( contextual0 ) ) # 将差值的绝对值，转为一个维度
    contextual_loss = tf.reduce_sum( contextual1, 1 ) # 将所有节点的值求和

    # 合并2个loss值
    complete_loss = contextual_loss + 0.99 * perceptual_loss
    
    # 梯度下降法求最佳值
    grad_z = tf.gradients(complete_loss, var_z) 
    
    # 更新z值
    learn_z = tf.constant(value=learning_rate, dtype=tf.float32) # 设置学习率
    z_opt = var_z.assign(var_z - learn_z * tf.reshape(grad_z, [noise_size])) # 将 1x100 转换为 100
    
    return z_opt, layer_z, complete_loss


    
    

