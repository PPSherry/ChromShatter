#!/usr/bin/env python
import pandas as pd
import os
import shutil
import random

# 设置随机种子以确保可重复性
random.seed(42)

# 定义路径
input_tsv = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.PCAWG/CNV_info_from_PDF/PCAWG_info_merge.tsv"
merge_dir = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.PCAWG/SV_graph.merge"
model_data_dir = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/model_data"

# 创建输出目录
train_dir = os.path.join(model_data_dir, "train")
test_dir = os.path.join(model_data_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 读取数据
df = pd.read_csv(input_tsv, sep='\t')

# 随机打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 计算测试集大小（1/10）
test_size = len(df) // 10

# 分割数据
test_df = df.iloc[:test_size]
train_df = df.iloc[test_size:]

# 保存分割后的数据
test_df.to_csv(os.path.join(model_data_dir, "test_data.tsv"), sep='\t', index=False)
train_df.to_csv(os.path.join(model_data_dir, "train_data.tsv"), sep='\t', index=False)

# 复制图片文件
def copy_images(df, target_dir):
    for img_path in df['image_path']:
        src_path = os.path.join(merge_dir, img_path)
        dst_path = os.path.join(target_dir, img_path)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)

# 复制测试集和训练集的图片
copy_images(test_df, test_dir)
copy_images(train_df, train_dir)

print(f"数据已分割并保存到 {model_data_dir}")
print(f"测试集大小: {len(test_df)}")
print(f"训练集大小: {len(train_df)}")
