# train_fine_tuned.py
import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, load_model
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

# 加载预训练的特征提取器模型
print("加载特征提取器模型作为微调基础...")
model_path = os.path.join(base_dir, 'model_A_feature_extractor.h5')
if not os.path.exists(model_path):
    print("错误：找不到特征提取器模型！请先运行 train_feature_extractor.py")
    exit()

model = load_model(model_path)

# 解冻VGG16基模型的一部分进行微调
base_model = model.layers[0]
base_model.trainable = True

# 只微调最后几个卷积块（block4和block5）
for layer in base_model.layers:
    if layer.name.startswith('block5') or layer.name.startswith('block4'):
        layer.trainable = True
        print(f"解冻层以便微调: {layer.name}")
    else:
        layer.trainable = False

# 重新编译模型，使用更小的学习率
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# 设置回调函数
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    ModelCheckpoint(
        filepath=os.path.join(base_dir, 'model_B_fine_tuned.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# 进行微调训练
print("开始微调训练...")
history = model.fit(
      train_generator,
      steps_per_epoch=train_generator.samples // batch_size,
      epochs=15,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples // batch_size,
      callbacks=callbacks,
      verbose=1)

# 绘制训练历史
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Fine-Tuned Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Fine-Tuned Model Loss')
plt.legend()
plt.savefig(os.path.join(base_dir, 'fine_tuned_training_history.png'))
plt.show()

print("微调模型训练完成！保存为 'model_B_fine_tuned.h5'")