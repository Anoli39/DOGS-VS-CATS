import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# 1. 加载训练好的模型
print("正在加载训练好的模型...")
model_path = r'C:\Users\25947\catdog_project\dog_vs_cat_cnn_model.h5'
model = load_model(model_path)
print("模型加载成功！")

# 2. 定义类别标签（根据训练时生成的索引）
# 注意：这个顺序必须和训练时flow_from_directory生成的class_indices一致！
# 您之前打印的 class_indices 应该是 {'cats': 0, 'dogs': 1} 或 {'dogs': 0, 'cats': 1}
# 如果不确定，可以运行训练脚本开头部分查看，或者根据概率解释（>0.5为狗）
class_labels = {0: '猫 (Cat)', 1: '狗 (Dog)'} # 这是我们假设的顺序，最常见的是0=猫，1=狗

# 3. 创建一个循环，让您可以持续预测图片
while True:
    print("\n" + "="*50)
    img_path = input("请输入图片的完整路径（或输入 'quit' 退出）: ")
    
    if img_path.lower() == 'quit':
        print("再见！")
        break
        
    # 检查文件是否存在
    if not os.path.isfile(img_path):
        print(f"错误：找不到文件 '{img_path}'")
        continue
        
    try:
        # 4. 加载和预处理图像
        # 目标尺寸必须与训练时一致 (150, 150)
        img = image.load_img(img_path, target_size=(150, 150))
        
        # 将图像转换为数组并添加批次维度
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # 形成 (1, 150, 150, 3) 的形状
        
        # 非常重要：使用与训练相同的归一化方式
        img_array /= 255.0
        
        # 5. 进行预测
        prediction = model.predict(img_array, verbose=0) # verbose=0不显示预测进度
        
        # 6. 解释预测结果
        # 因为我们使用sigmoid激活函数，输出是一个介于0和1之间的值
        # 我们可以理解为"是狗的概率"
        dog_probability = prediction[0][0]
        
        # 根据阈值（通常为0.5）决定类别
        if dog_probability > 0.5:
            predicted_class = 1 # 狗
            confidence = dog_probability
        else:
            predicted_class = 0 # 猫
            confidence = 1 - dog_probability
        
        # 7. 显示结果
        print(f"\n预测结果: 这是一只 {class_labels[predicted_class]}")
        print(f"置信度: {confidence:.2%}") # 格式化为百分比，保留两位小数
        print(f"原始输出值: {dog_probability:.6f}")
        
        # 额外解释
        if dog_probability > 0.5:
            print(f"(模型认为有{dog_probability:.2%}的可能性是狗)")
        else:
            print(f"(模型认为有{1-dog_probability:.2%}的可能性是猫)")
            
    except Exception as e:
        print(f"处理图像时发生错误: {e}")