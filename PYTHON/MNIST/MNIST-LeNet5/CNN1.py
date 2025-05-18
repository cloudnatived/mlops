import time
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets( 'Minist_data',one_hot=True)

#参数初始化
input_num = 784 # 输入的列数
labels = 10 #输出的列数
batchsize = 128 #训练集每一批次的照片
max_epochs = 1000 #迭代的次数
dropout = 0.85 

#这里设置的x,y的作用是来存储输入的照片个数，和标签个数
x = tf.placeholder(tf.float32,[None, input_num])
y = tf.placeholder(tf.float32,[None, labels])

# 数据处理，标注化 
def normallize( x ):
    mean_x = np.mean( x )
    std_x = np.std( x )
    x = (x - mean_x)/ std_x
    return x
#设置卷积层，x:输入的照片，w对应的权值，这里才去的是不填充
def con2d(x , w , b, strides = 1):
    x = tf.nn.conv2d(x, w, strides=[1, strides,strides,1], padding = 'VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu( x)
#池化层
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides =[1, k, k, 1] , padding = 'SAME')


#设置模型
def con2dnet(x, weights, biases ,dropout):
    #因为输入的数据是1行784行，需要转化为28行，28列，这里只是对于一张图开始讨论的哈
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    #第一个卷积层  28x28x1 change to 24x24x32
    con_1 = con2d( x, weights['wc1'] , biases['bd1'])
     #第一个池化层 24x24x32 change to 12x12x32
    con_1_maxpol = maxpool2d(con_1, k=2)
     #第二个卷积层 12x12x32 change to 8x8x64
    con_2 = con2d( con_1_maxpol, weights['wc2'] , biases['bd2'])
     #第二个池化层 8x8x64 change to 4x4x64
    con_1_maxpo2 = maxpool2d(con_2, k=2)
    #全连接层 4*4*64(每一个特征图4*4，共有64个)，变化成一行4*4*64，便于全连接
    #这里批次是128张图，那么就是128个行4*4*64,功能如同下面代码二的：layers.flatten()
    fc1 = tf.reshape(con_1_maxpo2,[-1,weight['wd1'].get_shape().as_list()[0]])
    #这个就是全连接层的计算 [1,4x4x64] change to [1, 1024] 
    fc2 = tf.add(tf.matmul(fc1, weight['wd1']), biases['bd3'])
    fc2 = tf.nn.relu(fc2)
    # dropout层
    fc3 = tf.nn.dropout(fc2,dropout)
    # [1,1024] change to [1, 10] 
    fc3 = tf.add(tf.matmul(fc2, weight['wd2']),biases['bd4'])
    return  fc3

#设置卷积层1,2对应的卷积核的大小，这里都是5x5 通过这里你会发现
#其实每个卷积核都不一样，这样的目的是提取不同方向维度的特征值
#这里wd1,wd2是两个全连接层对应的权值，类似于神经网络的正向传递
weight = {'wc1':tf.Variable(tf.random_normal([5,5,1,32])),

          'wc2':tf.Variable(tf.random_normal([5,5,32,64])),

          'wd1':tf.Variable(tf.random_normal([4*4*64,1024])),

          'wd2':tf.Variable(tf.random_normal([1024,10]))}

#这里是网络层的  y = wx + b
biases = {'bd1':tf.Variable(tf.random_normal([32])),

          'bd2':tf.Variable(tf.random_normal([64])),
          
          'bd3':tf.Variable(tf.random_normal([1024])),
          
          'bd4':tf.Variable(tf.random_normal([10]))}

pred = con2dnet( x, weight, biases , dropout)
coss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(coss)

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
corrct_num = tf.reduce_sum(tf.cast(correct_prediction, "float"))
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    accucy_list = [] #存续每次迭代后的准确率
    accucy_coss = [] #存续每次迭代后的损失值率
    sess.run(init_op)
    for eopch in range(max_epochs):
        train_x, train_y = mnist.train.next_batch(batchsize) 
        z = sess.run(optimizer, feed_dict={x: train_x, y: train_y})
        coss_1, num_1 = sess.run([coss, corrct_num], feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('epoch:{0}, accucy:{1}:'.format(eopch, num_1/10000))
        accucy_list.append(num_1/10000)
        accucy_coss.append(coss_1/10000)
        
    plt.title('test_accucy')
    plt.xlabel('epochs')
    plt.ylabel('accucy')
    plt.plot(accucy_list) 
    plt.show()
    
    plt.title('test_coss')
    plt.xlabel('epochs')
    plt.ylabel('coss')
    plt.plot(accucy_coss) 
    plt.show()
    

