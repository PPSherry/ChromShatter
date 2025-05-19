#!/usr/bin/env python
import pandas as pd
import os
import shutil
import random
from sklearn.model_selection import train_test_split

# 设置随机种子以确保可重复性
random.seed(42)

# 定义路径
input_tsv = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/model_extended_data/merge_data.tsv" # table contains all the data
merge_dir = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/model_extended_data/graph_dir" # directory contains all the graph files
model_data_dir = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/model_extended_data" # directory to save the training and testing data

# 创建输出目录
train_dir = os.path.join(model_data_dir, "train")
test_dir = os.path.join(model_data_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 读取数据
df = pd.read_csv(input_tsv, sep='\t')

# 显示原始数据中各label分类的数量
print("原始数据标签分布:")
print(df['label'].value_counts())
print(f"总数据量: {len(df)}")

# 按照label列进行分层采样，测试集比例为1/10
test_size = 0.1
train_df, test_df = train_test_split(
    df, 
    test_size=test_size, 
    random_state=42, 
    stratify=df['label']
)

# 显示分割后的训练集和测试集中各label的数量
print("\n训练集标签分布:")
print(train_df['label'].value_counts())
print(f"训练集总数: {len(train_df)}")

print("\n测试集标签分布:")
print(test_df['label'].value_counts())
print(f"测试集总数: {len(test_df)}")

# 计算训练集和测试集中各label的比例
train_label_ratio = train_df['label'].value_counts(normalize=True)
test_label_ratio = test_df['label'].value_counts(normalize=True)

print("\n比例比较:")
for label in sorted(df['label'].unique()):
    print(f"Label {label} - 训练集: {train_label_ratio.get(label, 0):.2%}, 测试集: {test_label_ratio.get(label, 0):.2%}")

# 保存分割后的数据
test_df.to_csv(os.path.join(model_data_dir, "test_data.tsv"), sep='\t', index=False)
train_df.to_csv(os.path.join(model_data_dir, "train_data.tsv"), sep='\t', index=False)

# 复制图片文件
def copy_images(df, target_dir):
    success_count = 0
    missing_count = 0
    
    for img_path in df['image_path']:
        # 确保目标目录存在
        os.makedirs(os.path.join(target_dir, os.path.dirname(img_path)), exist_ok=True)
        
        src_path = os.path.join(merge_dir, img_path)
        dst_path = os.path.join(target_dir, img_path)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            success_count += 1
        else:
            print(f"警告: 未找到图片 {src_path}")
            missing_count += 1
    
    return success_count, missing_count

# 复制测试集和训练集的图片
print("\n复制训练集图片...")
train_success, train_missing = copy_images(train_df, train_dir)
print(f"复制测试集图片...")
test_success, test_missing = copy_images(test_df, test_dir)

print(f"\n数据已分割并保存到 {model_data_dir}")
print(f"训练集: {len(train_df)}个样本, 图片复制成功{train_success}个, 缺失{train_missing}个")
print(f"测试集: {len(test_df)}个样本, 图片复制成功{test_success}个, 缺失{test_missing}个")
