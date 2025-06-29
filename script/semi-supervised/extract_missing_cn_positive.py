#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys

# 配置路径
BASE_DIR = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model"
INPUT_FILE = os.path.join(BASE_DIR, "SV_graph.simulated/all_samples_results.tsv")
OUTPUT_FILE = os.path.join(BASE_DIR, "SV_graph.simulated/missing_cn_positive_cases.tsv")

def main():
    """筛选缺失cn_2和cn_3且为shatterSeek阳性的行"""
    print("="*60)
    print("筛选缺失cn_2和cn_3且为shatterSeek阳性的案例")
    print("="*60)
    
    # 加载数据
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 文件 {INPUT_FILE} 不存在")
        sys.exit(1)
    
    df = pd.read_csv(INPUT_FILE, sep='\t')
    print(f"原始数据总行数: {len(df)}")
    
    # 筛选条件
    # 1. max_number_oscillating_CN_segments_2_states和max_number_oscillating_CN_segments_3_states均缺失
    missing_cn_condition = df['max_number_oscillating_CN_segments_2_states'].isna() & df['max_number_oscillating_CN_segments_3_states'].isna()
    # 2. chromothripsis_status为阳性（High Confidence或Low Confidence）
    positive_condition = df['chromothripsis_status'].isin(['High Confidence', 'Low Confidence'])
    
    # 组合条件
    final_condition = missing_cn_condition & positive_condition
    
    # 筛选数据
    filtered_df = df[final_condition]
    
    print(f"\n=== 筛选结果 ===")
    print(f"缺失cn_2和cn_3的总行数: {missing_cn_condition.sum()}")
    print(f"chromothripsis阳性的总行数: {positive_condition.sum()}")
    print(f"同时满足两个条件的行数: {len(filtered_df)}")
    
    if len(filtered_df) > 0:
        # 保存筛选结果
        filtered_df.to_csv(OUTPUT_FILE, sep='\t', index=False)
        print(f"\n筛选结果已保存到: {OUTPUT_FILE}")
        
        # 显示一些基本统计信息
        print(f"\n=== 筛选数据统计 ===")
        if 'number_CNV_segments' in filtered_df.columns:
            cn_seg_values = filtered_df['number_CNV_segments']
            if not cn_seg_values.isna().all():
                print(f"number_CNV_segments取值范围: {cn_seg_values.min():.0f} - {cn_seg_values.max():.0f}")
                print(f"number_CNV_segments平均值: {cn_seg_values.mean():.1f}")
        
        # 显示前几行数据的关键列
        key_columns = ['case_id', 'chrom', 'start', 'end', 'max_number_oscillating_CN_segments_2_states', 'max_number_oscillating_CN_segments_3_states', 'number_CNV_segments', 'chromothripsis_status']
        available_columns = [col for col in key_columns if col in filtered_df.columns]
        
        print(f"\n=== 前5行关键信息 ===")
        print(filtered_df[available_columns].head())
        
    else:
        print("\n警告: 没有找到满足条件的数据行")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main() 