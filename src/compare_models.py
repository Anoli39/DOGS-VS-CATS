# compare_models.py - 修复版
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置 matplotlib 使用支持英文的默认字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置路径
base_dir = r'C:\Users\25947\catdog_project'

# 加载三个模型
print("Loading three models...")
try:
    model_original = load_model(os.path.join(base_dir, 'dog_vs_cat_cnn_model.h5'))
    model_feature = load_model(os.path.join(base_dir, 'model_A_feature_extractor.h5'))
    model_fine_tuned = load_model(os.path.join(base_dir, 'model_B_fine_tuned.h5'))
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# 统一的图片预处理函数
def prepare_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# 创建模型比较函数
def predict_and_compare(model, img_path, model_name, target_size):
    """使用指定模型预测图片并返回结果"""
    processed_img = prepare_image(img_path, target_size)
    prediction = model.predict(processed_img, verbose=0)[0][0]
    
    if prediction > 0.5:
        label = "Dog"
        confidence = prediction
    else:
        label = "Cat"
        confidence = 1 - prediction
        
    return {
        'model': model_name,
        'prediction': label,
        'confidence': f"{confidence:.2%}",
        'raw_value': prediction
    }

# 主比较循环
while True:
    print("\n" + "="*60)
    img_path = input("Enter the image path (or type 'quit' to exit): ").strip('"')
    
    if img_path.lower() == 'quit':
        break
        
    if not os.path.isfile(img_path):
        print(f"Error: File not found '{img_path}'")
        continue
        
    try:
        # 为不同模型准备不同尺寸的输入
        # 原始模型使用 (150, 150)，VGG模型使用 (224, 224)
        results = []
        results.append(predict_and_compare(model_original, img_path, "Original Model", (150, 150)))
        results.append(predict_and_compare(model_feature, img_path, "Feature Extractor", (224, 224)))
        results.append(predict_and_compare(model_fine_tuned, img_path, "Fine-Tuned Model", (224, 224)))
        
        # 创建图形
        fig, (ax_img, ax_table) = plt.subplots(2, 1, figsize=(10, 10))
        fig.suptitle('Model Comparison Results', fontsize=16, fontweight='bold')
        
        # 上图：显示图片
        img = image.load_img(img_path)
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title('Input Image', fontsize=14, pad=10)
        
        # 下图：创建结果表格
        ax_table.axis('off')
        ax_table.axis('tight')
        
        # 准备表格数据
        table_data = []
        for result in results:
            table_data.append([
                result['model'],
                result['prediction'],
                result['confidence']
            ])
        
        # 创建表格
        table = ax_table.table(
            cellText=table_data,
            colLabels=['Model', 'Prediction', 'Confidence'],
            loc='center',
            cellLoc='center'
        )
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # 设置表头样式
        for i in range(3):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 为数据行设置样式
        for i in range(1, 4):
            # 设置行背景色
            color = '#f5f5f5' if i % 2 == 0 else 'white'
            for j in range(3):
                table[(i, j)].set_facecolor(color)
            
            # 高亮显示高置信度结果
            confidence = float(table_data[i-1][2].strip('%'))
            if confidence > 90:
                table[(i, 2)].set_facecolor('#c6efce')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()
        
        # 同时在控制台输出结果
        print("\n" + "="*40)
        print("MODEL COMPARISON RESULTS:")
        print("="*40)
        for result in results:
            print(f"{result['model']:20} -> {result['prediction']:5} (Confidence: {result['confidence']})")
            
    except Exception as e:
        print(f"Error processing image: {e}")