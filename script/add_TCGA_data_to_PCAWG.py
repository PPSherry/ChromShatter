#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, Image
import sys

# 基础路径
BASE_DIR = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model"
ALL_SAMPLES_RESULTS = os.path.join(BASE_DIR, "SV_graph.simulated/all_samples_results.tsv")
MERGE_DATA = os.path.join(BASE_DIR, "label_model_data/merge_data.tsv")
SV_GRAPH_DIR = os.path.join(BASE_DIR, "SV_graph.simulated/TCGA_graph")  # 存放原始图片的目录
TARGET_GRAPH_DIR = os.path.join(BASE_DIR, "label_model_data/graph_dir")  # 存放选中图片的目录

# 设置筛选条件 - 可以根据需要修改
# 格式: {列名: (比较操作, 值)}
# 比较操作可以是: '==', '!=', '>', '<', '>=', '<=', 'in', 'contains'
filter_conditions = {
    "chromothripsis_status": ("in", ["High Confidence", "Low Confidence"]),
    # 例如: "number_CNV_segments": (">", 5),
    # 例如: "plot_path": ("contains", ".png")
}

# 创建目标图片目录
os.makedirs(TARGET_GRAPH_DIR, exist_ok=True)

# 映射关系；
# TCGA -> PCAWG
mapping = {
    "case_id": "sample_name",
    "chrom": "Chr",
    "start": "Start",
    "end": "End",
    "max_number_oscillating_CN_segments_2_states": "cn_2",
    "max_number_oscillating_CN_segments_3_states": "cn_3",
    "number_CNV_segments": "cn_segments",
    "plot_path": "image_path"
}

def load_data():
    """加载数据集"""
    try:
        all_samples_df = pd.read_csv(ALL_SAMPLES_RESULTS, sep='\t')
        merge_data_df = pd.read_csv(MERGE_DATA, sep='\t')
        return all_samples_df, merge_data_df
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        sys.exit(1)

def filter_data(df, conditions):
    """根据条件筛选数据"""
    filtered_df = df.copy()
    
    for column, (operator, value) in conditions.items():
        if column not in filtered_df.columns:
            print(f"警告: 列 '{column}' 不存在于数据中, 跳过此筛选条件")
            continue
            
        if operator == "==":
            filtered_df = filtered_df[filtered_df[column] == value]
        elif operator == "!=":
            filtered_df = filtered_df[filtered_df[column] != value]
        elif operator == ">":
            filtered_df = filtered_df[filtered_df[column] > value]
        elif operator == "<":
            filtered_df = filtered_df[filtered_df[column] < value]
        elif operator == ">=":
            filtered_df = filtered_df[filtered_df[column] >= value]
        elif operator == "<=":
            filtered_df = filtered_df[filtered_df[column] <= value]
        elif operator == "in":
            filtered_df = filtered_df[filtered_df[column].isin(value)]
        elif operator == "contains":
            filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(value)]
    
    return filtered_df

def show_image(image_path):
    """显示图片"""
    try:
        full_path = os.path.join(SV_GRAPH_DIR, image_path)
        if os.path.exists(full_path):
            img = mpimg.imread(full_path)
            plt.figure(figsize=(12, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            return True
        else:
            print(f"警告: 图片 {full_path} 不存在")
            return False
    except Exception as e:
        print(f"显示图片时出错: {str(e)}")
        return False

def map_row_to_pcawg_format(row):
    """将all_samples_results的行映射为PCAWG格式"""
    pcawg_row = {}
    
    # 应用映射关系
    for tcga_col, pcawg_col in mapping.items():
        if tcga_col in row.index:
            pcawg_row[pcawg_col] = row[tcga_col]
    
    # 添加label列(用户输入的判断)
    pcawg_row['label'] = None  # 将由用户输入填充
    
    # 添加shatterSeek_label列(根据chromothripsis_status判断)
    if 'chromothripsis_status' in row.index:
        if row['chromothripsis_status'] in ['High Confidence', 'Low Confidence']:
            pcawg_row['shatterSeek_label'] = 1
        else:
            pcawg_row['shatterSeek_label'] = 0
    
    return pcawg_row

def copy_image(src_path, dest_path):
    """复制图片文件"""
    try:
        shutil.copy2(src_path, dest_path)
        return True
    except Exception as e:
        print(f"复制图片时出错: {str(e)}")
        return False

def main():
    print("加载数据...")
    all_samples_df, merge_data_df = load_data()
    
    print(f"原始数据共 {len(all_samples_df)} 行")
    print(f"应用筛选条件: {filter_conditions}")
    
    filtered_df = filter_data(all_samples_df, filter_conditions)
    print(f"筛选后数据共 {len(filtered_df)} 行")
    
    if len(filtered_df) == 0:
        print("没有找到符合条件的数据")
        return
    
    # 统计已处理的行数和追加的行数
    processed_count = 0
    appended_count = 0
    
    # 逐行处理筛选后的数据
    for idx, row in filtered_df.iterrows():
        processed_count += 1
        
        print("\n" + "="*80)
        print(f"处理第 {processed_count}/{len(filtered_df)} 行:")
        
        # 显示行的关键信息
        print(f"case_id: {row['case_id']}, chrom: {row['chrom']}")
        print(f"chromothripsis_status: {row['chromothripsis_status']}")
        print(f"cn_2: {row['max_number_oscillating_CN_segments_2_states']}", "cn_3: {row['max_number_oscillating_CN_segments_3_states']}", "cn_segments: {row['number_CNV_segments']}")
        print(f"cn_segments: {row['number_CNV_segments']}")
        print(f"clusterSize: {row['clusterSize']}")
        print(f"Plot: {row['plot_path']}")
        
        # 尝试显示图片
        image_path = row['plot_path']
        image_exists = show_image(image_path)
        
        if not image_exists:
            print("无法显示图片，跳过此行")
            continue
        
        # 获取用户判断
        while True:
            user_input = input("\n请判断此图是否为染色体碎裂事件 (1=是/真阳性, 0=否/假阳性, -1=无法判断, q=退出): ").strip()
            
            if user_input.lower() == 'q':
                print("用户退出")
                return
            
            try:
                judgment = int(user_input)
                if judgment in [1, 0, -1]:
                    break
                else:
                    print("输入无效，请输入 1, 0 或 -1")
            except ValueError:
                print("输入无效，请输入 1, 0 或 -1")
        
        # 如果用户无法判断，则跳过此行
        if judgment == -1:
            print("用户无法判断，跳过此行")
            continue
        
        # 映射行数据到PCAWG格式
        pcawg_row = map_row_to_pcawg_format(row)
        pcawg_row['label'] = judgment
        
        # 计算shatterSeek_label_TorF
        pcawg_row['shatterSeek_label_TorF'] = 1 if pcawg_row['label'] == pcawg_row['shatterSeek_label'] else 0
        
        # 复制图片到目标目录
        if image_exists:
            src_image_path = os.path.join(SV_GRAPH_DIR, image_path)
            dest_image_path = os.path.join(TARGET_GRAPH_DIR, image_path)
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
            
            if copy_image(src_image_path, dest_image_path):
                print(f"已复制图片到 {dest_image_path}")
            else:
                print(f"复制图片失败")
        
        # 将行追加到merge_data.tsv
        new_row_df = pd.DataFrame([pcawg_row])
        merge_data_df = pd.concat([merge_data_df, new_row_df], ignore_index=True)
        
        # 保存更新后的merge_data.tsv
        merge_data_df.to_csv(MERGE_DATA, sep='\t', index=False)
        print(f"已将行追加到 {MERGE_DATA}")
        
        appended_count += 1
    
    print("\n" + "="*80)
    print(f"处理完成, 共处理 {processed_count} 行, 追加 {appended_count} 行到 {MERGE_DATA}")

if __name__ == "__main__":
    main()


