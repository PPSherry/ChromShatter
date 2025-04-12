# 从PCAWG提供Supplementary Table 1中提取异常的call (False Postive/False Negative)
import pandas as pd
import os

# 文件路径
input_file = "/Users/xurui/back_up_unit/天津大学文件/本科毕设相关/Article/PCAWG-SupplementTable1.xlsx"  # 替换为你的 .xlsx 文件路径
output_file = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/PCAWG-related/abnormal_calls_from_PCAWG.tsv"  # 替换为输出的 .tsv 文件路径

try:
    df = pd.read_excel(input_file)
    
    # 过滤包含非空comment的行
    if 'comment' in df.columns:
        filtered_df = df[df['comment'].notna() & (df['comment'] != '')]
        filtered_df.to_csv(output_file, sep='\t', index=False)
        print(f"成功保存 {len(filtered_df)} 行到 {output_file}")
    else:
        print("Excel文件中未找到'comment'列。")

except FileNotFoundError:
    print(f"找不到文件: {input_file}")
except Exception as e:
    print(f"发生错误: {e}")