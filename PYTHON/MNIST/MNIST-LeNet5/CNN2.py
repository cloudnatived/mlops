import time
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 修复数据路径拼写错误（Minist_data → mnist_data）
mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# 参数初始化
input_num = 784    # 输入维度（28x28）
num_classes = 10   # 输出类别数
batch_size = 128   # 批次大小
max_epochs = 1000  # 最大迭代次数
dropout_rate = 0.85 # Dropout比率

# 数据预处理函数（标准化）
def normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std if std != 0 else x  # 避免除零错误

# 卷积层函数（带ReLU激活）
def conv2d(x, w, b, strides=1):
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# 最大池化层函数
def max_pool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# 构建CNN模型
def cnn_model(x, weights, biases, dropout):
    # 输入重塑：[None, 784] → [None, 28, 28, 1]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    
    # 第一层卷积+池化：28x28x1 → 24x24x32 → 12x12x32
    conv1 = conv2d(x, weights['wc1'], biases['b1'])
    pool1 = max_pool2d(conv1)
    pool1 = tf.nn.dropout(pool1, dropout)  # 新增卷积层后的Dropout
    
    # 第二层卷积+池化：12x12x32 → 8x8x64 → 4x4x64
    conv2 = conv2d(pool1, weights['wc2'], biases['b2'])
    pool2 = max_pool2d(conv2)
    pool2 = tf.nn.dropout(pool2, dropout)  # 新增池化层后的Dropout
    
    # 全连接层：4x4x64 → 1024
    fc1 = tf.reshape(pool2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['b3'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)  # 全连接层后的Dropout
    
    # 输出层：1024 → 10
    logits = tf.add(tf.matmul(fc1, weights['wd2']), biases['b4'])
    return logits

# 权重与偏置初始化（使用He初始化）
def he_init(shape):
    return tf.random_normal(shape, stddev=tf.sqrt(2.0 / shape[0]))  # 假设shape[0]为输入维度

weights = {
    'wc1': tf.Variable(he_init([5, 5, 1, 32])),   # 5x5卷积核，1输入通道，32输出通道
    'wc2': tf.Variable(he_init([5, 5, 32, 64])),  # 5x5卷积核，32输入通道，64输出通道
    'wd1': tf.Variable(he_init([4*4*64, 1024])),  # 全连接层1权重
    'wd2': tf.Variable(he_init([1024, 10]))       # 全连接层2权重
}

biases = {
    'b1': tf.Variable(tf.zeros([32])),            # 卷积层1偏置
    'b2': tf.Variable(tf.zeros([64])),            # 卷积层2偏置
    'b3': tf.Variable(tf.zeros([1024])),          # 全连接层1偏置
    'b4': tf.Variable(tf.zeros([10]))             # 输出层偏置
}

# 输入占位符
x = tf.placeholder(tf.float32, [None, input_num])
y = tf.placeholder(tf.float32, [None, num_classes])

# 构建模型
logits = cnn_model(x, weights, biases, dropout_rate)

# 损失函数与优化器
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# 评估指标
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化变量
init = tf.global_variables_initializer()

# 训练与评估
with tf.Session() as sess:
    sess.run(init)
    train_acc_list = []
    test_acc_list = []
    loss_list = []
    
    for epoch in range(max_epochs):
        # 训练批次
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # 标准化输入数据
        batch_x = normalize(batch_x)
        # 执行优化
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        
        # 每10个epoch评估一次
        if (epoch + 1) % 10 == 0:
            # 训练集准确率
            train_acc = sess.run(accuracy, feed_dict={
                x: normalize(mnist.train.images[:1000]),  # 取部分训练集评估
                y: mnist.train.labels[:1000]
            })
            # 测试集准确率与损失
            test_acc, current_loss = sess.run([accuracy, loss], feed_dict={
                x: normalize(mnist.test.images),
                y: mnist.test.labels
            })
            
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            loss_list.append(current_loss)
            
            print(f'Epoch {epoch+1}/{max_epochs}, '
                  f'Train Acc: {train_acc:.4f}, '
                  f'Test Acc: {test_acc:.4f}, '
                  f'Loss: {current_loss:.4f}')
    
    # 绘制准确率与损失曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs (每10次记录)')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(loss_list, label='Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs (每10次记录)')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
