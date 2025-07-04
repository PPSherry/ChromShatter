#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shatterSeek预测性能分析脚本

功能：
分析shatterSeek在手工标注数据上的预测性能
- 输入：semi_supervised_learning/manual_label/merge.tsv
- shatterSeek_label列：shatterSeek预测结果
- label列：真实标签
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

def analyze_shatterseek_performance():
    """分析shatterSeek预测性能"""
    print("=== shatterSeek 预测性能分析 ===")
    
    # 加载手工标注数据
    df = pd.read_csv('semi_supervised_learning/manual_label/merge.tsv', sep='\t')
    
    print(f"总样本数: {len(df)}")
    print(f"阳性样本数 (label=1): {(df['label'] == 1).sum()}")
    print(f"阴性样本数 (label=0): {(df['label'] == 0).sum()}")
    
    # 检查shatterSeek_label列的情况
    print(f"\nshatterSeek_label列情况:")
    print(f"非空值数量: {(~df['shatterSeek_label'].isna()).sum()}")
    print(f"空值数量: {df['shatterSeek_label'].isna().sum()}")
    
    if (~df['shatterSeek_label'].isna()).sum() > 0:
        print(f"shatterSeek_label取值情况:")
        print(df['shatterSeek_label'].value_counts(dropna=False))
    
    # 过滤掉shatterSeek_label或label为空的行
    valid_df = df.dropna(subset=['shatterSeek_label', 'label'])
    
    if len(valid_df) == 0:
        print("\n警告：没有有效的预测-标签对进行分析")
        print("shatterSeek_label列可能全部为空值")
        return None
    
    print(f"\n有效样本数: {len(valid_df)}")
    
    # 获取真实标签和预测标签
    y_true = valid_df['label'].astype(int)
    y_pred = valid_df['shatterSeek_label'].astype(int)
    
    print(f"真实标签分布:")
    print(f"  阳性: {(y_true == 1).sum()}")
    print(f"  阴性: {(y_true == 0).sum()}")
    
    print(f"shatterSeek预测分布:")
    print(f"  阳性: {(y_pred == 1).sum()}")
    print(f"  阴性: {(y_pred == 0).sum()}")
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tp + fp + tn + fn
    
    print(f"\n混淆矩阵:")
    print(f"True Negatives (TN): {tn} ({tn/total:.1%})")
    print(f"False Positives (FP): {fp} ({fp/total:.1%})") 
    print(f"False Negatives (FN): {fn} ({fn/total:.1%})")
    print(f"True Positives (TP): {tp} ({tp/total:.1%})")
    print(f"总计: {total}")
    
    # 计算性能指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n性能指标:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # 详细分类报告
    print(f"\n详细分类报告:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive'], zero_division=0))
    
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'accuracy': accuracy, 'precision': precision, 
        'recall': recall, 'f1': f1
    }

def main():
    """主函数"""
    print("开始shatterSeek预测性能分析...")
    print("=" * 50)
    
    try:
        results = analyze_shatterseek_performance()
        if results:
            print(f"\n=== 分析结果总结 ===")
            print(f"混淆矩阵: TP={results['tp']}, FP={results['fp']}, TN={results['tn']}, FN={results['fn']}")
            print(f"准确率 (Accuracy): {results['accuracy']:.4f}")
            print(f"精确率 (Precision): {results['precision']:.4f}")
            print(f"召回率 (Recall): {results['recall']:.4f}")
            print(f"F1分数 (F1-score): {results['f1']:.4f}")
        else:
            print("\n分析失败：无有效数据")
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 