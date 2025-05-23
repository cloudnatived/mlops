# 构建了一个基本的卷积神经网络（CNN）来训练 MNIST 手写数字识别任务，并带有训练曲线可视化。
#下面是你原始 CNN 模型的 TensorFlow 2.x / Keras 风格重写版本，保持了原有结构（两层卷积 + Dropout + 全连接），支持可视化准确率和损失曲线，并使用 He 初始化和标准化输入

# 1. 数据预处理：
#    采用标准化（z-score）而非简单归一化，更适合 CNN 训练
#    添加通道维度（expand_dims）以匹配模型输入要求
#    使用tf.data的cache()和prefetch()提升数据加载效率
# 2. 模型架构：
#    采用两次卷积 + 池化结构，逐步提取特征
#    使用 He 初始化（适合 ReLU 激活函数）加速收敛
#    多层 Dropout 抑制过拟合
# 3. 训练策略：
#    早停机制（EarlyStopping）防止过度训练
#    同时监控训练集和测试集指标，便于发现过拟合
# 4. 保存格式：
#    .keras：Keras 原生格式，推荐用于 TensorFlow 2.x
#    .h5：HDF5 格式，兼容旧版 Keras
#    model.export()：错误写法，应为model.save()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import os

# 超参数配置
INPUT_SHAPE = (28, 28, 1)           # 输入图像的尺寸：28×28像素，单通道（灰度图）
NUM_CLASSES = 10                    # 分类类别数（0-9数字）
BATCH_SIZE = 128                    # 每批次训练样本数
MAX_EPOCHS = 50                     # 最大训练轮次
DROPOUT_RATE = 0.15                 # Dropout比率，防止过拟合
MODEL_H5_PATH = "mnist_cnn.h5"      # HDF5模型保存路径（原代码，后续被覆盖）

# 数据预处理函数
def normalize_img(image, label):
    """将图像像素值归一化到[-1,1]范围，并对标签进行one-hot编码"""
    image = tf.cast(image, tf.float32) / 255.0                         # 转换为float32并归一化到[0,1]
    mean, var = tf.nn.moments(image, axes=[0, 1, 2])                   # 计算图像均值和方差
    std = tf.sqrt(var)                                                 # 计算标准差
    image = (image - mean) / (std + 1e-6)                              # 标准化（z-score），防止除零错误
    return tf.expand_dims(image, -1), tf.one_hot(label, NUM_CLASSES)   # 添加通道维度，标签转为one-hot

# 数据加载
def load_datasets():
    """加载MNIST数据集并进行预处理"""
    (ds_train, ds_test), _ = tfds.load(
        'mnist',                                                       # 加载MNIST手写数字数据集
        split=['train', 'test'],                                       # 分割为训练集和测试集
        shuffle_files=True,                                            # 随机打乱文件顺序
        as_supervised=True,                                            # 返回(input, label)元组格式
        with_info=True                                                 # 获取数据集元信息（未使用）
    )

    # 训练集处理：预处理、缓存、打乱、分批、预取
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # 测试集处理：预处理、分批、缓存、预取
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    
    return ds_train, ds_test

# 模型构建
def build_model():                                                                            # PyTorch：基于类的继承，需手动定义__init__（初始化层）和forward（定义数据流向）。
    """构建CNN模型架构"""
    model = tf.keras.Sequential([                                                             # Keras Sequential API：直接堆叠层，无需显式定义forward。
        # 第一个卷积块：32个5×5卷积核，ReLU激活，He初始化，有效填充
        tf.keras.layers.Conv2D(32, kernel_size=5, activation='relu', padding='valid',         # Keras 层自动初始化：建层（如Conv2D、Dense）时，权重和偏置会自动初始化。
                               kernel_initializer='he_normal', input_shape=INPUT_SHAPE),      # 可通过kernel_initializer参数指定初始化方法（如 He 初始化、Xavier 初始化）。
        tf.keras.layers.MaxPooling2D(pool_size=2),                                            # 最大池化，尺寸减半
        tf.keras.layers.Dropout(DROPOUT_RATE),                                                # 随机丢弃部分神经元，防止过拟合
        
        # 第二个卷积块：64个5×5卷积核
        tf.keras.layers.Conv2D(64, kernel_size=5, activation='relu', padding='valid',
                               kernel_initializer='he_normal'),
        tf.keras.layers.MaxPooling2D(pool_size=2),                                             # 最大池化
        tf.keras.layers.Dropout(DROPOUT_RATE),
        
        # 全连接层
        tf.keras.layers.Flatten(),                                                             # 将多维特征展平为一维
        tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal'),        # 1024个神经元
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')                               # 输出层，softmax激活，输出概率分布
    ])

    # 编译模型：配置优化器、损失函数和评估指标
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),                               # Adam优化器，学习率0.001
        loss='categorical_crossentropy',                                                       # 分类交叉熵损失（适用于one-hot标签）
        metrics=['accuracy']                                                                   # 准确率评估指标
    )
    return model

# 可视化训练过程
def plot_metrics(history):
    """绘制训练过程中的准确率和损失曲线"""
    plt.figure(figsize=(12, 5))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')     # 训练集准确率
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')  # 测试集准确率
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')     # 训练集损失
    plt.plot(history.history['val_loss'], label='Test Loss')  # 测试集损失
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()                                     # 自动调整子图布局
    plt.show()

# 主函数
def main():
    """主程序入口：加载数据、训练模型、可视化结果、保存模型"""
    ds_train, ds_test = load_datasets()                    # 加载并预处理数据集
    model = build_model()                                  # 构建CNN模型
    
    # 训练模型
    history = model.fit(
        ds_train,                                          # 训练数据集
        epochs=MAX_EPOCHS,                                 # 最大训练轮次
        validation_data=ds_test,                           # 验证数据集
        callbacks=[
                                                           # 早停回调：当验证集损失5轮未改善时停止训练，并恢复最佳权重
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )
    
    # 可视化训练结果
    plot_metrics(history)
    
    # 保存模型（存在重复保存问题）
    if not os.path.exists(MODEL_DIR):                      # 检查目录是否存在（原代码注释，未使用）
        os.makedirs(MODEL_DIR)
    
    # 以下为重复保存操作，实际使用时应选择合适的格式
    model.save("CNN1_saved_model.keras")                   # 保存为Keras格式（推荐）
    model.save("CNN1_mnist_model.h5")                      # 保存为HDF5格式
    model.export("CNN1_saved_model")                       # 错误：应为model.save()，此处会报错
    
    print(f"模型已保存到 CNN1_saved_model.keras")

# 入口点
if __name__ == '__main__':
    main()
