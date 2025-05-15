#!/usr/bin/env python
# merge CNV_info_from_PDF dataset
import pandas as pd
import os
from collections import Counter

def merge_and_check_duplicates():
    """
    合并四个数据集并检查复合键是否有重复
    """
    try:
        # 设置基础路径
        base_path = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.PCAWG/CNV_info_from_PDF"
        
        # 定义文件路径和对应的来源标签
        file_info = [
            {"path": os.path.join(base_path, "HC_split.tsv"), "source": "HC"},
            {"path": os.path.join(base_path, "LC_split.tsv"), "source": "LC"},
            {"path": os.path.join(base_path, "D3_split.tsv"), "source": "D3"},
            {"path": os.path.join(base_path, "D4_split.tsv"), "source": "D4"}
        ]
        
        # 存储所有数据框的列表
        all_dfs = []
        
        # 读取每个文件
        for info in file_info:
            file_path = info["path"]
            source_label = info["source"]
            
            if not os.path.exists(file_path):
                print(f"警告: 文件 {file_path} 不存在，已跳过")
                continue
                
            print(f"正在读取 {file_path}...")
            df = pd.read_csv(file_path, sep='\t')
            
            # 检查必要的列是否存在
            required_cols = ['sample_name', 'Chr', 'Start', 'End']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"警告: 文件 {file_path} 缺少以下列: {', '.join(missing_cols)}，已跳过")
                continue
            
            # 添加来源标识列
            df['source'] = source_label
            all_dfs.append(df)
            print(f"  读取了 {len(df)} 行数据，标记来源为 {source_label}")
        
        if not all_dfs:
            print("错误: 没有有效的数据文件")
            return
            
        # 合并所有数据框
        print("合并所有数据...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"合并后共有 {len(combined_df)} 行数据")
        
        # 确保数据类型一致
        combined_df['sample_name'] = combined_df['sample_name'].astype(str)
        combined_df['Chr'] = combined_df['Chr'].astype(str)
        combined_df['Start'] = pd.to_numeric(combined_df['Start'], errors='coerce')
        combined_df['End'] = pd.to_numeric(combined_df['End'], errors='coerce')
        
        # 创建复合键
        print("创建复合键...")
        combined_df['composite_key'] = combined_df.apply(
            lambda row: f"{row['sample_name']}_{row['Chr']}_{row['Start']}_{row['End']}", axis=1
        )
        
        # 按复合键排序
        print("按复合键排序...")
        combined_df = combined_df.sort_values(by='composite_key')
        
        # 输出合并后的所有数据（已排序）
        raw_merge_path = os.path.join(base_path, "raw_merge.tsv")
        combined_df.to_csv(raw_merge_path, sep='\t', index=False)
        print(f"已将合并后的 {len(combined_df)} 行数据保存到 {raw_merge_path}")
        
        # 计算每个键出现的次数
        key_counts = Counter(combined_df['composite_key'])
        
        # 找出重复的键
        duplicate_keys = {key: count for key, count in key_counts.items() if count > 1}
        
        if duplicate_keys:
            print(f"发现 {len(duplicate_keys)} 个重复的复合键")
            
            # 找出所有重复的行
            duplicate_rows = combined_df[combined_df['composite_key'].isin(duplicate_keys.keys())]
            
            # 保存重复项到文件
            duplicate_path = os.path.join(base_path, "duplicate_term.tsv")
            duplicate_rows.to_csv(duplicate_path, sep='\t', index=False)
            print(f"已将 {len(duplicate_rows)} 行重复数据保存到 {duplicate_path}")
            
            # 计算每个来源的重复情况
            source_counts = Counter(duplicate_rows['source'])
            print("各来源的重复行数:")
            for source, count in sorted(source_counts.items()):
                print(f"  {source}: {count}行 ({count/len(duplicate_rows)*100:.1f}%)")
        else:
            print("没有发现重复的复合键")
            
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")

if __name__ == "__main__":
    merge_and_check_duplicates()
