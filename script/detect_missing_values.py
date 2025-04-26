#!/usr/bin/env python
import pandas as pd
import os
import sys

def detect_missing_values(input_file, output_file, cleaned_output_file=None, remove_missing=False):
    """
    检测TSV文件中的缺失值，并将包含缺失值的行输出到新文件
    如果remove_missing为True，则将去除缺失值的数据保存到cleaned_output_file
    
    Args:
        input_file (str): 输入的TSV文件路径
        output_file (str): 输出的TSV文件路径，包含缺失值的行
        cleaned_output_file (str, optional): 输出的TSV文件路径，不包含缺失值的行
        remove_missing (bool, optional): 是否移除缺失值并保存到新文件
    """
    try:
        # 读取TSV文件
        df = pd.read_csv(input_file, sep='\t')
        
        # 找出包含缺失值的行
        missing_rows = df[df.isnull().any(axis=1)]
        
        if len(missing_rows) > 0:
            # 保存包含缺失值的行到新文件
            missing_rows.to_csv(output_file, sep='\t', index=False)
            
            # 打印统计信息
            print(f"Total rows in input file: {len(df)}")
            print(f"Rows with missing values: {len(missing_rows)}")
            print(f"Missing values by column:")
            for column in df.columns:
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    print(f"  {column}: {missing_count} missing values")
            
            print(f"\nResults saved to: {output_file}")
            
            # 如果需要移除缺失值，保存清洗后的数据
            if remove_missing and cleaned_output_file:
                # 获取不包含缺失值的行
                cleaned_df = df.dropna()
                # 保存到新文件
                cleaned_df.to_csv(cleaned_output_file, sep='\t', index=False)
                print(f"Cleaned data (with {len(cleaned_df)} rows) saved to: {cleaned_output_file}")
        else:
            print("No missing values found in the input file.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def main():
    # 设置输入输出文件路径
    input_file = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.PCAWG/CNV_info_from_PDF/dataset4_info/combined_events.tsv"
    output_file = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.PCAWG/CNV_info_from_PDF/dataset4_info/combined_events_missing_values.tsv"
    
    # 设置是否移除缺失值
    remove_missing_values = False
    
    # 如果需要移除缺失值，设置清洗后的输出文件路径
    cleaned_output_file = None
    if remove_missing_values:
        cleaned_output_file = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.PCAWG/CNV_info_from_PDF/dataset4_info/combined_events_remove_missing_values.tsv"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        sys.exit(1)
    
    # 执行缺失值检测
    detect_missing_values(input_file, output_file, cleaned_output_file, remove_missing_values)

if __name__ == "__main__":
    main()
