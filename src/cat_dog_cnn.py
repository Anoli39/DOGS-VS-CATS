import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 设置随机种子以保证结果可重现
np.random.seed(42)

# ！！！！！！ 定义关键路径 - 根据您的精确结构 ！！！！！！
base_dir = r'C:\Users\25947\catdog_project'
train_data_dir = os.path.join(base_dir, 'dogs-vs-cats', 'train') # 整理后的训练数据路径
test_data_dir = os.path.join(base_dir, 'dogs-vs-cats', 'test')   # Kaggle测试集路径

print(f"Training data path: {train_data_dir}")
print(f"Test data path: {test_data_dir}")

# 图像预处理和增强
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest',
      validation_split=0.2  # 直接从训练集中分20%作为验证集
)

# 验证数据只需要归一化，不需要增强
validation_datagen = ImageDataGenerator(
      rescale=1./255,
      validation_split=0.2
)

# 从目录生成训练数据批次
train_generator = train_datagen.flow_from_directory(
        train_data_dir,        # 目标目录
        target_size=(150, 150), # 调整所有图像为150x150
        batch_size=32,
        class_mode='binary',    # 二分类问题
        subset='training')      # 指定为训练子集

# 从目录生成验证数据批次
validation_generator = validation_datagen.flow_from_directory(
        train_data_dir,        # 注意：和训练集是同一个源目录
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='validation')    # 指定为验证子集

# 查看生成器的类别索引（0代表猫，1代表狗？或者反过来？）
print(f"Class indices: {train_generator.class_indices}")

# 构建卷积神经网络（CNN）模型
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 二分类输出层

model.summary()

# 编译模型
model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=1e-4),
              metrics=['accuracy'])

# 设置回调函数
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-7)

# 开始训练模型！
print("Starting training...")
history = model.fit(
      train_generator,
      steps_per_epoch=train_generator.samples // train_generator.batch_size,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples // validation_generator.batch_size,
      callbacks=[early_stop, reduce_lr],
      verbose=1)

# 训练完成后，保存模型
model_save_path = os.path.join(base_dir, 'dog_vs_cat_cnn_model.h5')
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# 可视化训练过程
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'training_history.png')) # 保存图表
plt.show()

print("All done!")