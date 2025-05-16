#!/usr/bin/env python
import pandas as pd
import os
import shutil
import random

# 设置随机种子以确保可重复性
random.seed(42)

# 定义路径
input_tsv = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/label_model_data/merge_data.tsv"
merge_graph_dir = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/label_model_data/graph_dir"

model_data_dir = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/label_model_data"
# 创建输出目录
train_dir = os.path.join(model_data_dir, "train")
test_dir = os.path.join(model_data_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 读取数据
df = pd.read_csv(input_tsv, sep='\t')

# 按照组合类型分组数据
tp_data = df[(df['shatterSeek_label'] == 1) & (df['shatterSeek_label_TorF'] == 1)]
fp_data = df[(df['shatterSeek_label'] == 1) & (df['shatterSeek_label_TorF'] == 0)]
tn_data = df[(df['shatterSeek_label'] == 0) & (df['shatterSeek_label_TorF'] == 1)]
fn_data = df[(df['shatterSeek_label'] == 0) & (df['shatterSeek_label_TorF'] == 0)]

# 计算各组数据总数
print(f"数据集分布情况:")
print(f"TP(True Positive): {len(tp_data)}")
print(f"FP(False Positive): {len(fp_data)}")
print(f"TN(True Negative): {len(tn_data)}")
print(f"FN(False Negative): {len(fn_data)}")

# 随机打乱每组数据
tp_data = tp_data.sample(frac=1, random_state=42).reset_index(drop=True)
fp_data = fp_data.sample(frac=1, random_state=43).reset_index(drop=True)
tn_data = tn_data.sample(frac=1, random_state=44).reset_index(drop=True)
fn_data = fn_data.sample(frac=1, random_state=45).reset_index(drop=True)

# 根据要求选择测试集
test_tp = tp_data.iloc[:20] if len(tp_data) >= 20 else tp_data
test_fp = fp_data.iloc[:16] if len(fp_data) >= 16 else fp_data
test_tn = tn_data.iloc[:10] if len(tn_data) >= 10 else tn_data
test_fn = fn_data.iloc[:8] if len(fn_data) >= 8 else fn_data

# 剩余数据作为训练集
train_tp = tp_data.iloc[20:] if len(tp_data) >= 20 else pd.DataFrame(columns=tp_data.columns)
train_fp = fp_data.iloc[16:] if len(fp_data) >= 16 else pd.DataFrame(columns=fp_data.columns)
train_tn = tn_data.iloc[10:] if len(tn_data) >= 10 else pd.DataFrame(columns=tn_data.columns)
train_fn = fn_data.iloc[8:] if len(fn_data) >= 8 else pd.DataFrame(columns=fn_data.columns)

# 合并测试集和训练集
test_df = pd.concat([test_tp, test_fp, test_tn, test_fn]).reset_index(drop=True)
train_df = pd.concat([train_tp, train_fp, train_tn, train_fn]).reset_index(drop=True)

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
        
        src_path = os.path.join(merge_graph_dir, img_path)
        dst_path = os.path.join(target_dir, img_path)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            success_count += 1
        else:
            print(f"警告: 未找到图片 {src_path}")
            missing_count += 1
    
    return success_count, missing_count

# 复制测试集和训练集的图片
print("复制测试集图片...")
test_copy_success, test_copy_missing = copy_images(test_df, test_dir)
print("复制训练集图片...")
train_copy_success, train_copy_missing = copy_images(train_df, train_dir)

print(f"\n数据已分割并保存到 {model_data_dir}")
print(f"测试集分布: TP={len(test_tp)}, FP={len(test_fp)}, TN={len(test_tn)}, FN={len(test_fn)}, 总计={len(test_df)}")
print(f"训练集分布: TP={len(train_tp)}, FP={len(train_fp)}, TN={len(train_tn)}, FN={len(train_fn)}, 总计={len(train_df)}")
print(f"图片复制情况: 测试集(成功={test_copy_success}, 缺失={test_copy_missing}), 训练集(成功={train_copy_success}, 缺失={train_copy_missing})")
