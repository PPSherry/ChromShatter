#!/usr/bin/env python
import pandas as pd

# 定义文件路径
input_file = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.simulated/SV_CNV_CaseID_table.tsv"
unique_output_file = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.simulated/SV_CNV_CaseID_table.unique.tsv"
duplicate_output_file = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.simulated/SV_CNV_CaseID_table.duplicate.tsv"

# 读取输入文件
df = pd.read_csv(input_file, sep='\t')
print(f"成功读取输入文件: {input_file}")
print(f"总行数: {len(df)}")

# 检查是否存在Case ID列
if 'Case ID' not in df.columns:
    print("错误: 输入文件中没有'Case ID'列")
    exit(1)

# 统计每个Case ID出现的次数
case_id_counts = df['Case ID'].value_counts()
duplicate_case_ids = case_id_counts[case_id_counts > 1].index.tolist()

print(f"唯一Case ID总数: {len(case_id_counts)}")
print(f"有重复的Case ID数量: {len(duplicate_case_ids)}")

# 保存每个Case ID第一次出现的行
unique_df = df.drop_duplicates(subset=['Case ID'], keep='first')

# 识别重复行
duplicate_df = df[df.duplicated(subset=['Case ID'], keep='first')]

# 保存结果
unique_df.to_csv(unique_output_file, sep='\t', index=False)
print(f"唯一Case ID的行已保存到: {unique_output_file}")
print(f"保存的行数: {len(unique_df)}")

if len(duplicate_df) > 0:
    duplicate_df.to_csv(duplicate_output_file, sep='\t', index=False)
    print(f"重复Case ID的行已保存到: {duplicate_output_file}")
    print(f"重复行数: {len(duplicate_df)}")
else:
    print("没有找到重复的Case ID行")
