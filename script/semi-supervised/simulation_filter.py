#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys

# 配置路径
BASE_DIR = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model"
ALL_SAMPLES_RESULTS = os.path.join(BASE_DIR, "SV_graph.simulated/all_samples_results.tsv")
MERGE_DATA = os.path.join(BASE_DIR, "semi_supervised_learning/manual_label/merge.tsv")
OUTPUT_FILE = os.path.join(BASE_DIR, "semi_supervised_learning/TCGA-test/simulation.remove_manual_label_part.tsv")

# 映射关系：ShatterSeek_output格式 -> PCAWG格式
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
    """加载数据文件"""
    print("正在加载数据文件...")
    
    # 加载TCGA数据
    if not os.path.exists(ALL_SAMPLES_RESULTS):
        print(f"错误: 文件 {ALL_SAMPLES_RESULTS} 不存在")
        sys.exit(1)
    
    tcga_df = pd.read_csv(ALL_SAMPLES_RESULTS, sep='\t')
    print(f"TCGA数据: {len(tcga_df)} 行")
    
    # 加载已标注数据
    if not os.path.exists(MERGE_DATA):
        print(f"错误: 文件 {MERGE_DATA} 不存在")
        sys.exit(1)
    
    merge_df = pd.read_csv(MERGE_DATA, sep='\t')
    print(f"已标注数据: {len(merge_df)} 行")
    
    return tcga_df, merge_df

def transform_tcga_data(tcga_df):
    """将TCGA数据转换为PCAWG格式"""
    print("正在转换数据格式...")
    
    # 检查必需的列是否存在
    missing_cols = [col for col in mapping.keys() if col not in tcga_df.columns]
    if missing_cols:
        print(f"警告: TCGA数据中缺少以下列: {missing_cols}")
    
    # 应用映射关系
    transformed_data = {}
    for tcga_col, pcawg_col in mapping.items():
        if tcga_col in tcga_df.columns:
            if pcawg_col in ['Start', 'End', 'cn_2', 'cn_3', 'cn_segments']:
                # 转换为数值类型
                transformed_data[pcawg_col] = pd.to_numeric(tcga_df[tcga_col], errors='coerce')
            else:
                transformed_data[pcawg_col] = tcga_df[tcga_col]
    
    # 创建DataFrame
    result_df = pd.DataFrame(transformed_data)
    
    # 添加空的标注列（与merge.tsv格式一致）
    result_df['label'] = None
    result_df['shatterSeek_label'] = None
    result_df['shatterSeek_label_TorF'] = None
    
    # 根据chromothripsis_status设置shatterSeek_label
    if 'chromothripsis_status' in tcga_df.columns:
        result_df['shatterSeek_label'] = tcga_df['chromothripsis_status'].apply(
            lambda x: 1 if x in ['High Confidence', 'Low Confidence'] else 0
        )
    
    print(f"转换后数据: {len(result_df)} 行")
    return result_df

def filter_data(transformed_df, merge_df):
    """筛选数据：只保留有image_path且未在merge.tsv中出现的行"""
    print("正在筛选数据...")
    
    # 筛选有image_path的行
    has_image = transformed_df.dropna(subset=['image_path'])
    print(f"有image_path的行: {len(has_image)} 行")
    
    # 获取已标注数据中的sample_name和Chr组合
    merge_df_clean = merge_df.dropna(subset=['sample_name', 'Chr'])
    existing_combinations = set(
        zip(merge_df_clean['sample_name'], merge_df_clean['Chr'].astype(str))
    )
    print(f"已标注的样本-染色体组合数: {len(existing_combinations)} 个")
    
    # 检查重复（基于sample_name和Chr的组合）
    has_image_clean = has_image.dropna(subset=['sample_name', 'Chr'])
    current_combinations = list(
        zip(has_image_clean['sample_name'], has_image_clean['Chr'].astype(str))
    )
    
    duplicate_mask = [combo in existing_combinations for combo in current_combinations]
    duplicate_indices = has_image_clean.index[duplicate_mask]
    duplicate_samples = has_image.index.isin(duplicate_indices)
    duplicate_count = duplicate_samples.sum()
    
    print(f"检测到重复样本-染色体组合: {duplicate_count} 个")
    
    if duplicate_count > 0:
        print("重复的样本-染色体组合（前10个）:")
        duplicate_combos = [(combo[0], combo[1]) for i, combo in enumerate(current_combinations) if duplicate_mask[i]][:10]
        for sample_name, chr_val in duplicate_combos:
            print(f"  - {sample_name} Chr{chr_val}")
        if len(duplicate_combos) == 10 and duplicate_count > 10:
            print(f"  ... 还有 {duplicate_count - 10} 个重复组合")
    
    # 排除重复样本
    filtered_df = has_image[~duplicate_samples].copy()
    print(f"筛选后数据: {len(filtered_df)} 行")
    
    return filtered_df, duplicate_count

def save_result(filtered_df):
    """保存结果到文件"""
    print("正在保存结果...")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_FILE)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存文件
    filtered_df.to_csv(OUTPUT_FILE, sep='\t', index=False)
    print(f"结果已保存到: {OUTPUT_FILE}")
    
    # 显示列信息
    print(f"输出文件包含列: {filtered_df.columns.tolist()}")
    print(f"输出文件行数: {len(filtered_df)}")

def main():
    """主函数"""
    print("="*80)
    print("ShatterSeek输出数据筛选脚本")
    print("目标: 创建未标注的ShatterSeek输出数据tsv表格")
    print("="*80)
    
    # 1. 加载数据
    tcga_df, merge_df = load_data()
    
    # 2. 转换数据格式
    transformed_df = transform_tcga_data(tcga_df)
    
    # 3. 筛选数据
    filtered_df, duplicate_count = filter_data(transformed_df, merge_df)
    
    # 4. 保存结果
    save_result(filtered_df)
    
    # 5. 总结
    print("\n" + "="*80)
    print("处理完成!")
    print(f"原始ShatterSeek输出数据: {len(tcga_df)} 行")
    print(f"有图像路径的行: {len(transformed_df.dropna(subset=['image_path']))} 行")
    print(f"与已标注数据重复: {duplicate_count} 行")
    print(f"最终输出数据: {len(filtered_df)} 行")
    print(f"输出文件: {OUTPUT_FILE}")
    print("="*80)

if __name__ == "__main__":
    main() 