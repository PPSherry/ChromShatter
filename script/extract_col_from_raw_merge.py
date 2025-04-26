#!/usr/bin/env python
import pandas as pd
import os
import numpy as np

def process_tsv_file(input_tsv, output_tsv):
    """处理TSV文件，按照指定要求转换数据并保存到新的TSV文件"""
    # 读取输入TSV文件
    df = pd.read_csv(input_tsv, sep='\t')
    
    # 将oscillating_cn_2&3_states拆分为cn_2和cn_3
    def split_cn(cn_str):
        if pd.isna(cn_str):
            return np.nan, np.nan
        try:
            parts = cn_str.split(',')
            if len(parts) >= 2:
                return parts[0].strip(), parts[1].strip()
            elif len(parts) == 1:
                return parts[0].strip(), parts[0].strip()
            else:
                return np.nan, np.nan
        except:
            return np.nan, np.nan
    
    # 应用拆分函数
    cn_split = df['oscillating_cn_2&3_states'].apply(split_cn)
    df['cn_2'] = [x[0] for x in cn_split]
    df['cn_3'] = [x[1] for x in cn_split]
    
    # 处理image_path：提取文件名，并加上source信息
    def process_image_path(row):
        if pd.isna(row['image_path']):
            return np.nan
        
        # 提取文件名
        basename = os.path.basename(row['image_path'])
        
        # 如果source存在，将其添加到文件名中
        if not pd.isna(row['source']):
            # 将source添加到文件名前面
            return f"{row['source']}_{basename}"
        else:
            return basename
    
    # 应用图像路径处理函数
    df['image_path'] = df.apply(process_image_path, axis=1)
    
    # 添加label列
    def assign_label(source):
        if pd.isna(source):
            return np.nan
        if source in ['HC', 'LC']:
            return 1
        elif source in ['D3', 'D4']:
            return 0
        else:
            return np.nan
    
    # 应用标签分配函数
    df['label'] = df['source'].apply(assign_label)
    
    # 选择并重排列
    result_df = df[['sample_name', 'Chr', 'Start', 'End', 'cn_2', 'cn_3', 'cn_segments', 'image_path', 'label']]
    
    # 保存到输出TSV文件
    result_df.to_csv(output_tsv, sep='\t', index=False)
    print(f"已处理并保存到 {output_tsv}")
    
    return result_df

if __name__ == "__main__":
    # 设置输入和输出文件路径
    input_tsv = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.PCAWG/CNV_info_from_PDF/raw_merge.tsv"
    output_tsv = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.PCAWG/CNV_info_from_PDF/PCAWG_info_merge.tsv"
    
    # 处理数据
    process_tsv_file(input_tsv, output_tsv)
