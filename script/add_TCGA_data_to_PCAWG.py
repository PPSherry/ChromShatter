#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TCGA数据添加到PCAWG数据集脚本

=== 脚本功能总结 ===
本脚本用于从TCGA染色体碎裂数据中筛选样本，通过人工标注的方式验证染色体碎裂事件，
并将标注后的数据整合到现有的PCAWG数据集中。

=== 主要功能 ===
1. 数据筛选：根据预设条件从TCGA数据中筛选候选样本
2. 图像展示：显示每个候选样本的SV图像供用户判断
3. 人工标注：用户手动标注每个样本是否为染色体碎裂事件
4. 数据整合：将标注后的数据转换为PCAWG格式并追加到现有数据集
5. 图像管理：将相关图像复制到统一的目录中
6. 去重处理：自动检测和移除重复条目

=== 使用方法 ===
1. 确保以下文件/目录存在：
   - SV_graph.simulated/all_samples_results.tsv (TCGA原始数据)
   - label_model_data/merge_data.tsv (现有PCAWG数据集)
   - SV_graph.simulated/TCGA_graph/ (TCGA图像目录)

 2. 修改筛选条件（第57-64行的filter_conditions）：
   - 可以设置CN分段数、聚类大小、染色体碎裂状态等筛选条件
   - 支持的比较操作：==, !=, >, <, >=, <=, in, contains

3. 运行脚本：
   python add_TCGA_data_to_PCAWG.py

4. 交互标注：
   - 对每个展示的图像进行判断
   - 输入1表示阳性（染色体碎裂）
   - 输入0表示阴性（非染色体碎裂）
   - 输入-1表示无法判断（跳过）
   - 输入q退出标注过程

=== 输出结果 ===
- 更新后的merge_data.tsv文件（包含新标注的数据）
- 复制的图像文件到graph_dir目录
- 处理统计信息（处理行数、追加行数、跳过重复项数等）

=== 注意事项 ===
- 脚本会自动去重，避免重复添加相同的数据条目
- 用户可随时输入'q'退出，已处理的数据会被保存
- 图像文件会被复制到统一目录便于后续使用
"""

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
SV_GRAPH_DIR = os.path.join(BASE_DIR, "SV_graph.simulated/TCGA_graph")  # 存放原始图片的目录

MERGE_DATA = os.path.join(BASE_DIR, "semi_supervised_learning/manual_label/merge.tsv")
TARGET_GRAPH_DIR = os.path.join(BASE_DIR, "semi_supervised_learning/manual_label/graph_dir")  # 存放选中图片的目录

# 设置筛选条件 - 可以根据需要修改
# 格式: {列名: (比较操作, 值)}
# 比较操作可以是: '==', '!=', '>', '<', '>=', '<=', 'in', 'contains'
filter_conditions = {
    "max_number_oscillating_CN_segments_2_states": ("<=", 3),
    "clusterSize": (">", 5),
    "chromothripsis_status": ("in", ["Not Significant"]),
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
            # 转换特定列为整数类型
            if pcawg_col in ['Start', 'End', 'cn_2', 'cn_3', 'cn_segments']:
                try:
                    pcawg_row[pcawg_col] = int(row[tcga_col])
                except (ValueError, TypeError):
                    print(f"警告: 无法将{tcga_col}({row[tcga_col]})转换为整数，保留原值")
                    pcawg_row[pcawg_col] = row[tcga_col]
            else:
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

def is_duplicate_entry(row, df):
    """检查行是否已存在于数据框中"""
    duplicate = df[
        (df['sample_name'] == row['sample_name']) & 
        (df['Chr'] == row['Chr']) & 
        (df['Start'] == row['Start']) & 
        (df['End'] == row['End'])
    ]
    return not duplicate.empty

def remove_duplicates(df):
    """移除重复的行，只保留第一个出现的行"""
    # 确保数据类型一致
    df['Chr'] = df['Chr'].astype(str)
    df['Start'] = pd.to_numeric(df['Start'], errors='coerce')
    df['End'] = pd.to_numeric(df['End'], errors='coerce')
    
    # 找出重复的行，只保留第一次出现的
    df_no_duplicates = df.drop_duplicates(subset=['sample_name', 'Chr', 'Start', 'End'], keep='first')
    
    # 如果有行被移除，打印信息
    dropped_rows = len(df) - len(df_no_duplicates)
    if dropped_rows > 0:
        print(f"移除了 {dropped_rows} 行重复数据")
    
    return df_no_duplicates

def main():
    print("加载数据...")
    all_samples_df, merge_data_df = load_data()
    
    # 首先检查并移除merge_data_df中的重复项
    print("检查并移除现有数据中的重复项...")
    original_len = len(merge_data_df)
    merge_data_df = remove_duplicates(merge_data_df)
    if len(merge_data_df) < original_len:
        # 如果移除了重复项，则保存更新后的文件
        merge_data_df.to_csv(MERGE_DATA, sep='\t', index=False)
        print(f"已将去重后的数据保存到 {MERGE_DATA}")
    
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
    skipped_duplicates = 0
    
    # 设置是否用户要求退出的标志
    user_quit = False
    
    # 逐行处理筛选后的数据
    for idx, row in filtered_df.iterrows():
        if user_quit:
            break
            
        processed_count += 1
        
        print("\n" + "="*80)
        print(f"处理第 {processed_count}/{len(filtered_df)} 行:")
        
        # 映射行数据到PCAWG格式（用于检查是否重复）
        pcawg_row = map_row_to_pcawg_format(row)
        
        # 检查是否重复
        if is_duplicate_entry(pcawg_row, merge_data_df):
            print(f"跳过重复项: {pcawg_row['sample_name']}, Chr={pcawg_row['Chr']}, Start={pcawg_row['Start']}, End={pcawg_row['End']}")
            skipped_duplicates += 1
            continue
        
        # 显示行的关键信息
        print(f"case_id: {row['case_id']}, chrom: {row['chrom']}")
        print(f"chromothripsis_status: {row['chromothripsis_status']}")
        print(f"    HC_standard: {row['HC_standard']}")
        print(f"    HC_supplement1: {row['HC_supplement1']}")
        print(f"    HC_supplement2: {row['HC_supplement2']}")
        print(f"    LC: {row['LC']}")
        print(f"cn_2: {row['max_number_oscillating_CN_segments_2_states']}", f"cn_3: {row['max_number_oscillating_CN_segments_3_states']}")
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
            user_input = input("\n请判断此图是否为染色体碎裂事件 (1=阳性, 0=阴性, -1=无法判断, q=退出): ").strip()
            
            if user_input.lower() == 'q':
                print("用户选择退出标注过程")
                user_quit = True
                break
            
            try:
                judgment = int(user_input)
                if judgment in [1, 0, -1]:
                    break
                else:
                    print("输入无效，请输入 1, 0 或 -1")
            except ValueError:
                print("输入无效，请输入 1, 0 或 -1")
        
        # 如果用户要求退出，跳过剩余处理
        if user_quit:
            continue
            
        # 如果用户无法判断，则跳过此行
        if judgment == -1:
            print("用户无法判断，跳过此行")
            continue
        
        # 设置用户输入的判断结果
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
    print(f"处理完成, 共处理 {processed_count} 行")
    print(f"跳过 {skipped_duplicates} 个重复项")
    print(f"追加 {appended_count} 行到 {MERGE_DATA}")
    
    if user_quit:
        print("用户中途退出，以上是已完成部分的汇总信息")
    
    # 最后再次检查并移除可能的重复项
    print("\n检查最终数据中的重复项...")
    final_df = pd.read_csv(MERGE_DATA, sep='\t')
    final_df_no_dup = remove_duplicates(final_df)
    if len(final_df_no_dup) < len(final_df):
        final_df_no_dup.to_csv(MERGE_DATA, sep='\t', index=False)
        print(f"已将最终去重后的数据保存到 {MERGE_DATA}")

if __name__ == "__main__":
    main()


