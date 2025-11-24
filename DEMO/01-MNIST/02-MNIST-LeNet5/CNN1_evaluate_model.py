import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 模型路径
MODEL_PATH = 'CNN1/best_model.h5'

# 输入形状和类别数
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10

# 数据预处理函数
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    mean, var = tf.nn.moments(image, axes=[0, 1, 2])
    std = tf.sqrt(var)
    image = (image - mean) / (std + 1e-6)
    return tf.expand_dims(image, -1), tf.one_hot(label, NUM_CLASSES)

def load_test_dataset():
    ds_test = tfds.load(
        'mnist',
        split='test',
        shuffle_files=False,
        as_supervised=True
    )
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128).prefetch(tf.data.AUTOTUNE)
    return ds_test

def evaluate(model_path):
    # 加载模型
    if os.path.isdir(model_path):
        model = tf.keras.models.load_model(model_path)  # SavedModel 格式
    elif os.path.isfile(model_path) and model_path.endswith('.h5'):
        model = tf.keras.models.load_model(model_path)  # HDF5 格式
    else:
        print(f"模型路径无效: {model_path}")
        sys.exit(1)

    # 加载测试集
    ds_test = load_test_dataset()

    # 模型评估
    loss, acc = model.evaluate(ds_test)
    print(f"\nTest Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    # 显示前5个预测样本
    raw_test = tfds.load('mnist', split='test', as_supervised=True).take(5)
    images, labels = [], []
    for img, label in raw_test:
        images.append(tf.expand_dims(img, -1))
        labels.append(label.numpy())
    images = tf.stack(images)
    images = tf.cast(images, tf.float32) / 255.0
    mean, var = tf.nn.moments(images, axes=[1, 2, 3], keepdims=True)
    std = tf.sqrt(var)
    images = (images - mean) / (std + 1e-6)

    preds = model.predict(images)
    pred_labels = tf.argmax(preds, axis=1).numpy()

    # 可视化预测结果
    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(tf.squeeze(images[i]), cmap='gray')
        plt.title(f"Pred: {pred_labels[i]}\nTrue: {labels[i]}")
        plt.axis('off')
    plt.suptitle("MNIST Prediction Samples")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    evaluate(MODEL_PATH)
