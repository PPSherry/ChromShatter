#!/usr/bin/env python
import pandas as pd
import os
import sys

def detect_missing_values(input_file, output_file):
    """
    检测TSV文件中的缺失值，并将包含缺失值的行输出到新文件
    
    Args:
        input_file (str): 输入的TSV文件路径
        output_file (str): 输出的TSV文件路径，包含缺失值的行
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
        else:
            print("No missing values found in the input file.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def main():
    # 设置输入输出文件路径
    input_file = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/text_info_from_PDF/combined_events.tsv"
    output_file = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/text_info_from_PDF/combined_events_missing_values.tsv"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        sys.exit(1)
    
    # 执行缺失值检测
    detect_missing_values(input_file, output_file)

if __name__ == "__main__":
    main()
