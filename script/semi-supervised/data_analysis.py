#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys

# 配置路径
BASE_DIR = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model"
DATA_FILE = os.path.join(BASE_DIR, "semi_supervised_learning/TCGA-test/simulation.remove_manual_label_part.tsv")

def main():
    """主函数"""
    print("="*60)
    print("TCGA测试数据分析")
    print("="*60)
    
    # 加载数据
    if not os.path.exists(DATA_FILE):
        print(f"错误: 文件 {DATA_FILE} 不存在")
        sys.exit(1)
    
    df = pd.read_csv(DATA_FILE, sep='\t')
    print(f"数据总行数: {len(df)}")
    
    # 1. 缺失值分析
    print(f"\n=== 缺失值分析 ===")
    cn_2_missing = df['cn_2'].isna().sum()
    cn_3_missing = df['cn_3'].isna().sum()
    cn_2_and_3_missing = (df['cn_2'].isna() & df['cn_3'].isna()).sum()
    cn_segments_missing = df['cn_segments'].isna().sum()
    
    print(f"cn_2列缺失: {cn_2_missing} 行 ({cn_2_missing/len(df)*100:.1f}%)")
    print(f"cn_3列缺失: {cn_3_missing} 行 ({cn_3_missing/len(df)*100:.1f}%)")
    print(f"cn_2列和cn_3列均缺失: {cn_2_and_3_missing} 行 ({cn_2_and_3_missing/len(df)*100:.1f}%)")
    print(f"cn_segments列缺失: {cn_segments_missing} 行 ({cn_segments_missing/len(df)*100:.1f}%)")
    
    # 2. cn_segments取值范围（针对缺失cn_2或cn_3的行）
    if cn_segments_missing == 0:
        missing_cn_rows = df[df['cn_2'].isna() | df['cn_3'].isna()]
        if len(missing_cn_rows) > 0:
            cn_seg_values = missing_cn_rows['cn_segments']
            print(f"\n缺失cn_2或cn_3的行中，cn_segments取值范围: {cn_seg_values.min():.0f} - {cn_seg_values.max():.0f}")
            if cn_seg_values.nunique() <= 20:
                print(f"所有唯一值: {sorted(cn_seg_values.unique())}")
    
    # 3. shatterSeek_label分布分析
    print(f"\n=== shatterSeek_label分布 ===")
    label_1_count = (df['shatterSeek_label'] == 1).sum()
    label_0_count = (df['shatterSeek_label'] == 0).sum()
    total_valid = label_1_count + label_0_count
    
    print(f"标签为1: {label_1_count} 行 ({label_1_count/total_valid*100:.1f}%)")
    print(f"标签为0: {label_0_count} 行 ({label_0_count/total_valid*100:.1f}%)")
    print(f"阳性率: {label_1_count/total_valid*100:.1f}%")
    
    # 4. 缺失cn_2,cn_3行的shatterSeek_label分布
    missing_cn_rows = df[df['cn_2'].isna() | df['cn_3'].isna()]
    if len(missing_cn_rows) > 0:
        print(f"\n=== 缺失cn_2或cn_3行的shatterSeek_label分布 ===")
        missing_label_1 = (missing_cn_rows['shatterSeek_label'] == 1).sum()
        missing_label_0 = (missing_cn_rows['shatterSeek_label'] == 0).sum()
        missing_total = missing_label_1 + missing_label_0
        
        print(f"缺失cn_2或cn_3的总行数: {len(missing_cn_rows)}")
        if missing_total > 0:
            print(f"  标签为1: {missing_label_1} 行 ({missing_label_1/missing_total*100:.1f}%)")
            print(f"  标签为0: {missing_label_0} 行 ({missing_label_0/missing_total*100:.1f}%)")
            print(f"  阳性率: {missing_label_1/missing_total*100:.1f}%")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main() 