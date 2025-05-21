# 构建了一个基本的卷积神经网络（CNN）来训练 MNIST 手写数字识别任务，并带有训练曲线可视化。
#下面是你原始 CNN 模型的 TensorFlow 2.x / Keras 风格重写版本，保持了原有结构（两层卷积 + Dropout + 全连接），支持可视化准确率和损失曲线，并使用 He 初始化和标准化输入
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import os

# 超参数配置
INPUT_SHAPE = (28, 28, 1)           # 输入的列数
NUM_CLASSES = 10                    
BATCH_SIZE = 128                    # 训练集每一批次的照片
MAX_EPOCHS = 50                     # 迭代的次数
DROPOUT_RATE = 0.15                 # Dropout比率
MODEL_DIR = "saved_model"
MODEL_H5_PATH = "mnist_cnn.h5"

# 数据预处理函数
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    mean, var = tf.nn.moments(image, axes=[0, 1, 2])
    std = tf.sqrt(var)
    image = (image - mean) / (std + 1e-6)
    return tf.expand_dims(image, -1), tf.one_hot(label, NUM_CLASSES)

# 数据加载
def load_datasets():
    (ds_train, ds_test), _ = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test

# 模型构建
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=5, activation='relu', padding='valid',
                               kernel_initializer='he_normal', input_shape=INPUT_SHAPE),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(DROPOUT_RATE),

        tf.keras.layers.Conv2D(64, kernel_size=5, activation='relu', padding='valid',
                               kernel_initializer='he_normal'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(DROPOUT_RATE),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 可视化训练过程
def plot_metrics(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 主函数
def main():
    ds_train, ds_test = load_datasets()
    model = build_model()

    history = model.fit(
        ds_train,
        epochs=MAX_EPOCHS,
        validation_data=ds_test,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

    # 可视化
    plot_metrics(history)

    # 保存模型
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    #model.save(MODEL_DIR)         # 保存为 TensorFlow SavedModel 格式
    #model.save(MODEL_H5_PATH)     # 另存为 HDF5 (.h5) 格式
    model.save("CNN1_saved_model.keras")
    model.save("CNN1_mnist_model.h5")
    model.export("CNN1_saved_model") 
    #print(f"模型已保存到 {MODEL_DIR} 和 {MODEL_H5_PATH}")
    print(f"模型已保存到 CNN1_saved_model.keras")

# 入口点
if __name__ == '__main__':
    main()
