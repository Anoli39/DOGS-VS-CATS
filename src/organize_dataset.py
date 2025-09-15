import os
import shutil

# 定义路径 - 根据您提供的精确路径
# 项目根目录：C:\Users\25947\catdog_project
base_dir = r'C:\Users\25947\catdog_project' # 使用原始字符串避免转义符问题
source_train_dir = os.path.join(base_dir, 'dogs-vs-cats', 'train') # 源训练目录
dest_cats_dir = os.path.join(source_train_dir, 'cats') # 目标：train/cats/
dest_dogs_dir = os.path.join(source_train_dir, 'dogs') # 目标：train/dogs/

# 创建目标文件夹
os.makedirs(dest_cats_dir, exist_ok=True)
os.makedirs(dest_dogs_dir, exist_ok=True)

print(f"Source directory: {source_train_dir}")
print(f"Moving cat images to: {dest_cats_dir}")
print(f"Moving dog images to: {dest_dogs_dir}")

# 计数器
moved_cats = 0
moved_dogs = 0

# 遍历源训练目录中的所有文件
for filename in os.listdir(source_train_dir):
    file_path = os.path.join(source_train_dir, filename)
    
    # 只处理文件，忽略目录（比如刚创建的cats和dogs文件夹）
    if os.path.isfile(file_path):
        if filename.startswith('cat.'):
            destination = os.path.join(dest_cats_dir, filename)
            shutil.move(file_path, destination)
            moved_cats += 1
            if moved_cats % 1000 == 0: # 每移动1000张打印一次进度
                print(f"... Moved {moved_cats} cat images so far...")
        elif filename.startswith('dog.'):
            destination = os.path.join(dest_dogs_dir, filename)
            shutil.move(file_path, destination)
            moved_dogs += 1
            if moved_dogs % 1000 == 0: # 每移动1000张打印一次进度
                print(f"... Moved {moved_dogs} dog images so far...")
        else:
            print(f'[Warning] File "{filename}" was skipped. It does not start with "cat." or "dog.".')

print("\nData organization completed!")
print(f"Total cat images moved: {moved_cats}")
print(f"Total dog images moved: {moved_dogs}")