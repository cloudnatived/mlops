# 构建了一个基本的卷积神经网络（CNN）来训练 MNIST 手写数字识别任务，并带有训练曲线可视化。
# 下面是原始 CNN 模型的 TensorFlow 2.x / Keras 风格重写版本，保持了原有结构（两层卷积 + Dropout + 全连接），支持可视化准确率和损失曲线，并使用 He 初始化和标准化输入
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
MODEL_DIR = "CNN1"

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
def build_optimized_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=INPUT_SHAPE),   # 👈 明确输入形状

        # 数据增强层
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomZoom(0.1),

        # 第一层卷积块
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal', input_shape=INPUT_SHAPE),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(DROPOUT_RATE),

        # 第二层卷积块
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(DROPOUT_RATE),

        # 全连接层
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE+0.1),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # 学习率衰减
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
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
    plt.savefig('CNN1/training_metrics.png')  # 保存图像
    plt.show()

# 主函数
def main():
    ds_train, ds_test = load_datasets()
    model = build_optimized_model()

    # 定义回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join('CNN1/logs', 'mnist_cnn'))
    ]

    # 确保模型保存目录存在
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    history = model.fit(
        ds_train,
        epochs=MAX_EPOCHS,
        validation_data=ds_test,
        callbacks=callbacks,
        verbose=1
    )

    # 可视化
    plot_metrics(history)

    # 保存最终模型
    model.save(os.path.join(MODEL_DIR, 'final_model.keras'))
    model.save(os.path.join(MODEL_DIR, 'final_model.h5'))
    model.save(os.path.join(MODEL_DIR, 'saved_model'))

    print(f"模型已保存到 {MODEL_DIR}")

# 入口点
if __name__ == '__main__':
    main()
