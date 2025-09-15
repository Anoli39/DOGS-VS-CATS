# train_feature_extractor.py
import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# 设置路径
base_dir = r'C:\Users\25947\catdog_project'
train_data_dir = os.path.join(base_dir, 'dogs-vs-cats', 'train')

# 图像预处理和数据增强
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      validation_split=0.2
)

validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# VGG16的标准输入尺寸是 (224, 224)
img_size = (224, 224)
batch_size = 32

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

validation_generator = validation_datagen.flow_from_directory(
        train_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation')

print(f"Class indices: {train_generator.class_indices}")

# 加载VGG16作为特征提取器（权重冻结）
print("加载VGG16作为特征提取器...")
base_model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

# 冻结所有预训练层的权重
base_model.trainable = False

# 在预训练模型基础上构建新模型
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# 设置回调函数
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint(
        filepath=os.path.join(base_dir, 'model_A_feature_extractor.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# 训练模型
print("开始训练特征提取器模型...")
history = model.fit(
      train_generator,
      steps_per_epoch=train_generator.samples // batch_size,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples // batch_size,
      callbacks=callbacks,
      verbose=1)

# 绘制训练历史
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Feature Extractor Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Feature Extractor Model Loss')
plt.legend()
plt.savefig(os.path.join(base_dir, 'feature_extractor_training_history.png'))
plt.show()

print("特征提取器模型训练完成！保存为 'model_A_feature_extractor.h5'")