# 从pdf_region_selector.py中提取的区域信息中提取文本

import fitz  # PyMuPDF
import os
import argparse
import importlib.util
import pandas as pd

def load_regions_from_file(regions_file):
    """从Python文件加载区域信息"""
    spec = importlib.util.spec_from_file_location("regions_module", regions_file)
    regions_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regions_module)
    
    return regions_module.pdf_path, regions_module.regions

def extract_text_from_regions(pdf_path, regions):
    """从PDF的指定区域提取文本"""
    # 打开PDF
    doc = fitz.open(pdf_path)
    
    # 存储结果
    results = []
    
    # 遍历每一页
    for page_num in sorted(regions.keys()):
        page_idx = page_num - 1  # 页码从1开始，索引从0开始
        
        if page_idx >= len(doc):
            print(f"警告: 页码 {page_num} 超出PDF范围")
            continue
            
        page = doc[page_idx]
        page_regions = regions[page_num]
        
        # 提取每个区域的文本
        for region_idx, region in enumerate(page_regions):
            top_left = region['top_left']
            bottom_right = region['bottom_right']
            
            # 创建矩形区域
            rect = fitz.Rect(top_left[0], top_left[1], bottom_right[0], bottom_right[1])
            
            # 提取区域内的文本
            text = page.get_text("text", clip=rect)
            
            # 去除多余空白
            text = ' '.join(text.split())
            
            # 从文本中识别某些关键信息
            sample_name = None
            position = None
            oscillating_cn = None
            cn_segments = None
            
            # 尝试识别位置信息
            if "Position" in text:
                position_parts = text.split("Position")
                if len(position_parts) > 1:
                    position_text = position_parts[1].strip()
                    # 尝试提取形如 "1:120690998−248630158" 的文本
                    import re
                    position_match = re.search(r'(\d+:\d+[−-]\d+)', position_text)
                    if position_match:
                        position = position_match.group(1)
            
            # 尝试识别震荡CN值
            if "Oscillating CN" in text:
                cn_parts = text.split("Oscillating CN")
                if len(cn_parts) > 1:
                    cn_text = cn_parts[1].strip()
                    # 尝试提取形如 "21, 25" 的文本
                    import re
                    cn_match = re.search(r'(\d+,\s*\d+)', cn_text)
                    if cn_match:
                        oscillating_cn = cn_match.group(1)
            
            # 尝试识别CN片段数
            if "CN segments" in text:
                segments_parts = text.split("CN segments")
                if len(segments_parts) > 1:
                    segments_text = segments_parts[1].strip()
                    # 尝试提取数字
                    import re
                    segments_match = re.search(r'(\d+)', segments_text)
                    if segments_match:
                        cn_segments = segments_match.group(1)
            
            # 保存结果
            results.append({
                'page': page_num,
                'region': region_idx + 1,
                'text': text,
                'position': position,
                'oscillating_cn_2&3_states': oscillating_cn,
                'cn_segments': cn_segments,
                'sample_name': sample_name
            })
    
    doc.close()
    return results

def save_to_tsv(results, output_file):
    """将结果保存为TSV文件"""
    df = pd.DataFrame(results)
    df.to_csv(output_file, sep='\t', index=False)
    print(f"结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="从PDF指定区域提取文本")
    parser.add_argument("regions_file", help="包含区域信息的Python文件")
    parser.add_argument("--output", "-o", default=None, help="输出TSV文件路径")
    
    args = parser.parse_args()
    
    # 加载区域信息
    pdf_path, regions = load_regions_from_file(args.regions_file)
    
    # 如果未指定输出文件，使用默认名称
    if args.output is None:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        args.output = f"{pdf_name}_extracted.tsv"
    
    # 提取文本
    print(f"正在从 {pdf_path} 提取文本...")
    results = extract_text_from_regions(pdf_path, regions)
    
    # 保存结果
    save_to_tsv(results, args.output)
    
    # 打印提取的区域数
    print(f"已从 {len(regions)} 页PDF中提取 {len(results)} 个区域的文本")

if __name__ == "__main__":
    main() 