from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os
from datetime import datetime

def main():
    # 数据路径设置
    data_root = os.path.abspath(os.getcwd())
    #image_path = os.path.join(data_root, "flower_data")
    image_path = os.path.join(data_root, "./")
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
    
    # 检查路径是否存在
    assert os.path.exists(train_dir), f"无法找到训练数据目录: {train_dir}"
    assert os.path.exists(validation_dir), f"无法找到验证数据目录: {validation_dir}"
    
    # 创建保存权重的目录
    save_dir = os.path.join("save_weights", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    # 模型和训练参数
    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 30
    num_classes = 5
    
    # 增强的数据生成器
    train_image_generator = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    validation_image_generator = ImageDataGenerator(rescale=1./255)
    
    # 加载训练数据
    train_data_gen = train_image_generator.flow_from_directory(
        directory=train_dir,
        batch_size=batch_size,
        shuffle=True,
        target_size=(im_height, im_width),
        class_mode='categorical'
    )
    total_train = train_data_gen.n
    
    # 获取类别字典并保存为JSON
    class_indices = train_data_gen.class_indices
    inverse_dict = {v: k for k, v in class_indices.items()}
    with open(os.path.join(save_dir, 'class_indices.json'), 'w') as f:
        json.dump(inverse_dict, f, indent=4)
    
    # 加载验证数据
    val_data_gen = validation_image_generator.flow_from_directory(
        directory=validation_dir,
        batch_size=batch_size,
        shuffle=False,
        target_size=(im_height, im_width),
        class_mode='categorical'
    )
    total_val = val_data_gen.n
    
    print(f"使用 {total_train} 张图像进行训练，{total_val} 张图像进行验证。")
    
    # 构建模型
    from model import AlexNet_v1
    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=num_classes)
    model.summary()
    
    # 定义回调函数
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            #filepath=os.path.join(save_dir, 'best_model.h5'),
            filepath=os.path.join(save_dir, 'best_model.weights.h5'),  # 修改文件扩展名
            save_best_only=True,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001
        ),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(save_dir, 'logs'))
    ]
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    # 训练模型
    history = model.fit(
        x=train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size,
        callbacks=callbacks
    )
    
    # 保存最终模型
    model.save_weights(os.path.join(save_dir, 'final_model.h5'))
    
    # 评估模型
    test_loss, test_acc = model.evaluate(val_data_gen, verbose=2)
    print(f'验证集准确率: {test_acc:.2f}')
    
    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.legend()
    plt.title('训练和验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.legend()
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.show()
    
    # 保存训练历史
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history.history, f)

if __name__ == '__main__':
    main()
