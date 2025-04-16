#!/usr/bin/env python
import pandas as pd
import os
import re
import argparse

def extract_file_id(filename):
    """从文件名中提取file_id (以.分割的第一部分)"""
    return filename.split('.')[0]

def extract_case_id(case_id_str):
    """处理Case ID (格式为'A, B'，其中A和B相同时只保留一个)"""
    if not isinstance(case_id_str, str):
        return None
        
    parts = [p.strip() for p in case_id_str.split(',')]
    # 去除重复
    unique_parts = list(set(parts))
    return ', '.join(unique_parts)

def match_sv_cnv_files(tcga_list_path, sample_sheet_path, output_path):
    """
    匹配SV和CNV文件，并与样本信息表整合
    
    Args:
        tcga_list_path: TCGA文件列表路径
        sample_sheet_path: 样本信息表路径
        output_path: 输出文件路径
    """
    # 读取TCGA文件列表
    try:
        tcga_df = pd.read_csv(tcga_list_path, sep='\t')
        print(f"成功读取TCGA文件列表: {tcga_list_path}")
        print(f"共有 {len(tcga_df)} 条记录")
    except Exception as e:
        print(f"读取TCGA文件列表时出错: {str(e)}")
        return
    
    # 读取样本信息表
    try:
        sample_df = pd.read_csv(sample_sheet_path, sep='\t')
        print(f"成功读取SV样本信息表: {sample_sheet_path}")
        print(f"共有 {len(sample_df)} 条记录")
    except Exception as e:
        print(f"读取SV样本信息表时出错: {str(e)}")
        return
    
    # 过滤出SV文件
    sv_files = tcga_df[tcga_df['filename'].str.endswith('.wgs.BRASS.raw_structural_variation.vcf.gz')]
    print(f"找到 {len(sv_files)} 个SV文件")
    
    # 过滤出CNV文件
    cnv_files = tcga_df[tcga_df['filename'].str.endswith('.wgs.CaVEMan.raw_somatic_mutation.vcf.gz')]
    print(f"找到 {len(cnv_files)} 个CNV文件")
    
    # 提取file_id
    sv_files['file_id'] = sv_files['filename'].apply(extract_file_id)
    cnv_files['file_id'] = cnv_files['filename'].apply(extract_file_id)
    
    # 创建SV文件路径
    sv_files['SV_path'] = sv_files.apply(lambda row: f"{row['id']}/{row['filename']}", axis=1)
    
    # 创建CNV文件路径
    cnv_files['CNV_path'] = cnv_files.apply(lambda row: f"{row['id']}/{row['filename']}", axis=1)
    
    # 准备样本信息映射
    sample_map = {}
    for _, row in sample_df.iterrows():
        if 'File Name' in row and 'Case ID' in row:
            file_id = extract_file_id(row['File Name'])
            sample_map[file_id] = extract_case_id(row['Case ID'])
    
    print(f"从样本信息表中提取了 {len(sample_map)} 个样本ID映射")
    
    # 匹配SV和CNV文件
    matched_pairs = []
    
    for _, sv_row in sv_files.iterrows():
        file_id = sv_row['file_id']
        sv_path = sv_row['SV_path']
        
        # 查找同名的CNV文件
        matching_cnv = cnv_files[cnv_files['file_id'] == file_id]
        
        if len(matching_cnv) > 0:
            cnv_path = matching_cnv.iloc[0]['CNV_path']
            
            # 获取Case ID
            case_id = sample_map.get(file_id, "未知")
            
            matched_pairs.append({
                'Case ID': case_id,
                'SV_path': sv_path,
                'CNV_path': cnv_path
            })
    
    print(f"成功匹配 {len(matched_pairs)} 对SV和CNV文件")
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(matched_pairs)
    
    # 保存结果
    result_df.to_csv(output_path, sep='\t', index=False)
    print(f"结果已保存到 {output_path}")

def main():
    tcga_list_path = "/Volumes/T7-shield/CS-Bachelor-Thesis/TCGA-wgs/tcga_wgs_list.txt"
    sample_sheet_path = "/Volumes/T7-shield/CS-Bachelor-Thesis/TCGA-wgs/SV_sample_sheet.2025-04-16.tsv"
    output_path = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.simulated/SV_CNV_CaseID_table.tsv"

    match_sv_cnv_files(tcga_list_path, sample_sheet_path, output_path)

if __name__ == "__main__":
    main()

