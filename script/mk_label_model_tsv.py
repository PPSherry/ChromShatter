#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os

def add_shatterseek_columns():
    """
    为test_data.tsv添加两列：
    1. shatterSeek_label: 基于abnormal_calls_from_PCAWG.tsv中的标记，得到原来ShatterSeek的判断结果
    2. shatterSeek_label_TorF: 判断shatterSeek_label与原始label是否一致
    """
    # 文件路径
    base_dir = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model"
    abnormal_file = os.path.join(base_dir, "PCAWG-related/abnormal_calls_from_PCAWG.tsv")
    test_data_file = os.path.join(base_dir, "label_model_data/train_data.tsv") # 也可以是train_data.tsv path
    output_file = test_data_file  # 直接覆盖原文件
    
    # 读取数据
    print("读取abnormal数据...")
    abnormal_df = pd.read_csv(abnormal_file, sep='\t')
    print("读取test_data数据...")
    test_df = pd.read_csv(test_data_file, sep='\t')
    
    # 添加新列用于记录ShatterSeek结果
    test_df['shatterSeek_label'] = None
    # 添加一个临时列记录是否真正匹配到了abnormal记录
    test_df['_matched_abnormal'] = False
    
    # 确保数据类型正确
    abnormal_df['Chr'] = abnormal_df['Chr'].astype(str)
    abnormal_df['Start'] = pd.to_numeric(abnormal_df['Start'], errors='coerce')
    abnormal_df['End'] = pd.to_numeric(abnormal_df['End'], errors='coerce')
    
    test_df['Chr'] = test_df['Chr'].astype(str)
    test_df['Start'] = pd.to_numeric(test_df['Start'], errors='coerce')
    test_df['End'] = pd.to_numeric(test_df['End'], errors='coerce')
    
    # 处理test_df中的染色体位置信息
    for idx, row in test_df.iterrows():
        if idx % 100 == 0:
            print(f"处理测试数据行 {idx}/{len(test_df)}...")
        
        # 查找匹配的abnormal记录
        matches = abnormal_df[
            (abnormal_df['donor_idx'] == row['sample_name']) &
            (abnormal_df['Chr'] == str(row['Chr'])) &
            (abnormal_df['Start'] == row['Start']) &
            (abnormal_df['End'] == row['End'])
        ]
        
        if len(matches) > 1:
            print(f"警告: 行 {idx} 匹配到多个abnormal记录 ({len(matches)}条)")
            test_df.at[idx, '_matched_abnormal'] = True
        elif len(matches) == 1:
            test_df.at[idx, '_matched_abnormal'] = True
            if 'comment' in matches.columns:
                comment = matches.iloc[0]['comment']
                if pd.notna(comment):
                    # 根据comment设置shatterseek结果
                    if "False positive; manually removed" in comment:
                        test_df.at[idx, 'shatterSeek_label'] = 1
                    elif "False negative; manually included" in comment:
                        test_df.at[idx, 'shatterSeek_label'] = 0
                    else:
                        # 如果注释不符合预期格式，打印警告
                        print(f"警告: 未知的comment格式: {comment}")
    
    # 统计真正匹配到的abnormal记录数
    true_match_count = test_df['_matched_abnormal'].sum()
    print(f"匹配到abnormal记录的数量: {true_match_count}")
    
    # 对于未匹配的数据，使用label列的值
    test_df.loc[test_df['shatterSeek_label'].isna(), 'shatterSeek_label'] = test_df.loc[test_df['shatterSeek_label'].isna(), 'label']
    
    # 确保shatterSeek_label是整数类型
    test_df['shatterSeek_label'] = test_df['shatterSeek_label'].astype(int)
    
    # 添加shatterSeek_label_TorF列，比较shatterSeek_label与原始label是否一致
    test_df['shatterSeek_label_TorF'] = (test_df['shatterSeek_label'] == test_df['label']).astype(int)
    
    # 报告结果
    total_count = len(test_df)
    non_na_count = sum(~test_df['shatterSeek_label'].isna())
    print(f"总记录数: {total_count}")
    print(f"非空shatterSeek_label记录数: {non_na_count}")
    print(f"其中从abnormal匹配到的记录数: {true_match_count}")
    print(f"从label填充的记录数: {non_na_count - true_match_count}")
    print(f"ShatterSeek标签统计: ")
    print(test_df['shatterSeek_label'].value_counts())
    
    print("\nShatterSeek与原始标签一致性统计:")
    print(test_df['shatterSeek_label_TorF'].value_counts())
    match_rate = test_df['shatterSeek_label_TorF'].mean() * 100
    print(f"一致率: {match_rate:.2f}%")
    
    # 删除临时列
    test_df.drop('_matched_abnormal', axis=1, inplace=True)
    
    # 保存修改后的数据
    test_df.to_csv(output_file, sep='\t', index=False)
    print(f"已将更新后的数据保存到 {output_file}")

if __name__ == "__main__":
    add_shatterseek_columns()
