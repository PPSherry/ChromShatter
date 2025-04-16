#!/usr/bin/env python
import cv2
import os
import argparse
import pandas as pd
from tqdm import tqdm

def check_single_image(image_path):
    """检查单张图片的尺寸"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None
    
    height, width = img.shape[:2]  # img.shape返回(height, width, channels)
    print(f"分辨率: {width} x {height}")
    return {"filename": os.path.basename(image_path), "width": width, "height": height}

def check_image_folder(image_dir):
    """检查文件夹中所有图片的尺寸"""
    results = []
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
    
    print(f"发现 {len(image_files)} 个图片文件")
    
    for filename in tqdm(image_files):
        image_path = os.path.join(image_dir, filename)
        img = cv2.imread(image_path)
        if img is not None:
            height, width = img.shape[:2]
            results.append({
                "filename": filename,
                "width": width,
                "height": height,
                "path": image_path
            })
        else:
            print(f"无法读取图片: {filename}")
    
    return results

def print_statistics(results):
    """打印统计信息"""
    if not results:
        print("没有有效的图片数据")
        return
    
    widths = [r["width"] for r in results]
    heights = [r["height"] for r in results]
    
    print("\n统计信息:")
    print(f"图片总数: {len(results)}")
    print(f"宽度: 最小={min(widths)}, 最大={max(widths)}, 平均={sum(widths)/len(widths):.1f}")
    print(f"高度: 最小={min(heights)}, 最大={max(heights)}, 平均={sum(heights)/len(heights):.1f}")
    
    # 计算最常见的分辨率
    resolutions = {}
    for r in results:
        resolution = f"{r['width']}x{r['height']}"
        resolutions[resolution] = resolutions.get(resolution, 0) + 1
    
    # 找出最常见的分辨率
    common_resolutions = sorted(resolutions.items(), key=lambda x: x[1], reverse=True)[:3]
    print("\n最常见的分辨率:")
    for resolution, count in common_resolutions:
        print(f"{resolution}: {count}张图片 ({count/len(results)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="检查图片分辨率")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--image", help="单张图片的路径")
    group.add_argument("-d", "--directory", help="图片文件夹的路径")
    parser.add_argument("-o", "--output", help="输出TSV文件的路径")
    
    args = parser.parse_args()
    
    results = []
    
    if args.image:
        # 检查单张图片
        print(f"检查图片: {args.image}")
        result = check_single_image(args.image)
        if result:
            results.append(result)
    else:
        # 检查文件夹中的所有图片
        print(f"检查文件夹: {args.directory}")
        results = check_image_folder(args.directory)
    
    # 打印统计信息
    print_statistics(results)
    
    # 保存到TSV文件
    if args.output and results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, sep='\t', index=False)
        print(f"\n结果已保存到: {args.output}")

if __name__ == "__main__":
    main()
