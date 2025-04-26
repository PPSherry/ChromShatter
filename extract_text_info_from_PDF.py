import fitz  # PyMuPDF
import pandas as pd
import re
import os

def extract_text_by_page(pdf_path):
    """提取PDF每页的文本块，保留位置信息"""
    doc = fitz.open(pdf_path)
    pages_text = []

    # 遍历每一页
    for page_num in range(len(doc)):
        page = doc[page_num]
        # 提取页面文本，按块（block）获取
        blocks = page.get_text("dict")["blocks"]
        page_text = []

        # 按块提取文本，并记录位置信息
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:  # 忽略空文本
                            # 记录文本和位置
                            bbox = span["bbox"]  # (x0, y0, x1, y1)
                            page_text.append({
                                "text": text, 
                                "x0": bbox[0], 
                                "y0": bbox[1],
                                "x1": bbox[2],
                                "y1": bbox[3]
                            })

        # 按左上到右下的顺序排序
        page_text = sorted(page_text, key=lambda x: (x["y0"], x["x0"]))
        pages_text.append(page_text)

    doc.close()
    return pages_text

def get_fixed_quadrant_coordinates():
    """根据人工提取的坐标，返回固定的四个象限区域坐标"""
    # 根据high-confidence-test_regions.py中的数据计算平均坐标
    quadrants = [
        # 左上区域
        {
            'min_x': 50.0,  # 略小于最小的left_x以确保覆盖
            'max_x': 290.0, # 略大于最大的right_x
            'min_y': 265.0, # 略小于最小的top_y
            # 'max_y': 420.0  # 略大于最大的bottom_y
            'max_y': 410.0
        },
        # 右上区域
        {
            'min_x': 315.0,
            'max_x': 565.0,
            'min_y': 265.0,
            # 'max_y': 420.0
            'max_y': 410.0
        },
        # 左下区域
        {
            'min_x': 45.0,
            'max_x': 290.0,
            'min_y': 670.0,
            # 'max_y': 835.0
            'max_y': 850.0
        },
        # 右下区域
        {
            'min_x': 315.0,
            'max_x': 565.0,
            'min_y': 670.0,
            # 'max_y': 835.0
            'max_y': 850.0
        }
    ]
    return quadrants

def group_text_into_events(page_text, num_events=4):
    """将页面文本分组为不同的染色体破裂事件，使用固定坐标范围"""
    # 获取固定的四个象限坐标
    quadrants = get_fixed_quadrant_coordinates()
    
    # 初始化四个区域的文本列表
    events = [[] for _ in range(num_events)]
    
    # 将每个文本项分配到对应的区域
    for item in page_text:
        # 获取文本的中心点
        center_x = (item["x0"] + item["x1"]) / 2
        center_y = (item["y0"] + item["y1"]) / 2
        
        # 检查文本中心点在哪个象限
        for i, quadrant in enumerate(quadrants):
            if (quadrant["min_x"] <= center_x <= quadrant["max_x"] and
                quadrant["min_y"] <= center_y <= quadrant["max_y"]):
                events[i].append(item)
                break
    
    return events

def extract_event_info(event_text, donor_ids=None):
    """从事件文本中提取关键信息"""
    info = {
        "sample_name": None,
        "position": None,
        "oscillating_cn_2&3_states": None,
        "cn_segments": None,
        "full_text": " ".join(item["text"] for item in event_text)
    }
    
    # 按行组织文本项，使解析更容易
    lines = []
    current_line = []
    current_y = None
    
    # 按Y坐标对文本分组为行
    sorted_text = sorted(event_text, key=lambda x: (x["y0"], x["x0"]))
    for item in sorted_text:
        if current_y is None or abs(item["y0"] - current_y) < 10:  # 10是容差值
            current_line.append(item)
            current_y = item["y0"]
        else:
            if current_line:
                lines.append(sorted(current_line, key=lambda x: x["x0"]))
            current_line = [item]
            current_y = item["y0"]
    
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x["x0"]))
    
    # 存储oscillating_cn值的文本块
    oscillating_cn_value_block = None
    
    # 第一步：提取位置信息
    position_matches = []
    for item in event_text:
        position_match = re.search(r'([0-9XY]+:\d+[−-]\d+)', item["text"])
        if position_match:
            position_matches.append({
                "position": position_match.group(1),
                "y0": item["y0"]
            })
    
    # 如果有匹配结果，选择y坐标最小的作为position值
    if position_matches:
        # 按y0坐标排序并取第一个（最上方的）
        position_matches.sort(key=lambda x: x["y0"])
        info["position"] = position_matches[0]["position"]
    
    # 第二步：提取震荡CN值
    for i, line in enumerate(lines):
        line_text = " ".join(item["text"] for item in line)
        if "Oscillating CN" in line_text:
            cn_parts = line_text.split("Oscillating CN")
            if len(cn_parts) > 1:
                cn_text = cn_parts[1].strip()
                # 提取CN值，格式如 "21, 25" 或 "7, 11"
                cn_match = re.search(r'(\d+,\s*\d+)', cn_text)
                if cn_match:
                    info["oscillating_cn_2&3_states"] = cn_match.group(1)
                    # 找到震荡CN值的文本块
                    for item in line:
                        if re.search(r'\d+,\s*\d+', item["text"]):
                            oscillating_cn_value_block = item
                            break
    
    # 第三步：简化提取CN segments的逻辑
    # 首先判断是否提取到了震荡CN值
    if oscillating_cn_value_block and info["oscillating_cn_2&3_states"]:
        # 定义X坐标范围（震荡CN值的左右各扩大10）
        min_x = oscillating_cn_value_block["x0"] - 20
        max_x = oscillating_cn_value_block["x1"] + 20
        y_threshold = oscillating_cn_value_block["y1"]  # 震荡CN值的底部Y坐标
        
        # 找出Y坐标在震荡CN值下方，且X坐标在范围内的所有文本块
        candidates = []
        for item in sorted_text:
            if (item["y0"] > y_threshold and 
                min_x <= item["x0"] <= max_x and
                re.match(r'^\d+$', item["text"].strip())):
                candidates.append(item)
        
        # 选择Y坐标最接近震荡CN值的候选项
        if candidates:
            closest_item = min(candidates, key=lambda x: x["y0"] - y_threshold)
            info["cn_segments"] = closest_item["text"].strip()
    
    # 样本名称识别
    # 如果有donor_ids参考列表，直接检查所有文本
    if donor_ids:
        for item in event_text:
            text = item["text"].strip()
            # 规范化提取的文本
            normalized_text = text.replace('−', '-').replace('–', '-').replace('‐', '-')
            
            # 遍历donor_ids列表
            for donor_id in donor_ids:
                # 确保donor_id是字符串类型
                donor_id_str = str(donor_id) if not isinstance(donor_id, str) else donor_id
                
                # 规范化donor_id
                normalized_donor_id = donor_id_str.replace('−', '-').replace('–', '-').replace('‐', '-')
                
                # 比较规范化后的文本
                if normalized_text == normalized_donor_id:
                    info["sample_name"] = donor_id  # 保存原始donor_id
                    break
                
                # UUID特殊处理：如果看起来像UUID格式，则忽略所有短横线比较
                if re.search(r'[0-9a-f]{8}[-−–]', normalized_text) and re.search(r'[0-9a-f]{8}[-−–]', normalized_donor_id):
                    # 移除所有短横线后比较
                    text_no_dash = re.sub(r'[-−–]', '', normalized_text)
                    donor_no_dash = re.sub(r'[-−–]', '', normalized_donor_id)
                    if text_no_dash == donor_no_dash:
                        info["sample_name"] = donor_id
                        break
            
            if info["sample_name"]:
                break
    
    return info

def load_donor_ids(xlsx_path):
    """从PCAWG-SupplementTable1.xlsx加载donor_idx列的唯一值"""
    try:
        if not os.path.exists(xlsx_path):
            print(f"警告: 参考文件 {xlsx_path} 不存在，将不使用donor_idx参考")
            return None
            
        df = pd.read_excel(xlsx_path)
        if 'donor_idx' not in df.columns:
            print("警告: 参考文件中没有donor_idx列，将不使用donor_idx参考")
            return None
            
        donor_ids = df['donor_idx'].unique().tolist()
        print(f"已从参考文件加载 {len(donor_ids)} 个donor_idx值")
        return donor_ids
    except Exception as e:
        print(f"加载donor_ids时出错: {str(e)}")
        return None

def process_pdf_to_tsv(pdf_path, output_tsv, donor_ids=None):
    """处理PDF并提取所有染色体破裂事件信息到TSV文件"""
    # 提取每页文本
    pages_text = extract_text_by_page(pdf_path)
    
    # 处理每页，提取事件信息
    data = []
    for page_num, page_text in enumerate(pages_text):
        
        # 使用固定坐标范围将页面文本分组为4个事件
        events = group_text_into_events(page_text)
        
        # 提取每个事件的信息
        for event_idx, event_text in enumerate(events):
            if not event_text:  # 跳过空事件
                continue
                
            event_info = extract_event_info(event_text, donor_ids)
            
            # 添加页码和事件索引
            event_info["page"] = page_num + 1
            event_info["event"] = event_idx + 1  # 1: 左上, 2: 右上, 3: 左下, 4: 右下
            
            # 添加象限名称
            quadrant_names = ["Upper_Left", "Upper_Right", "Lower_Left", "Lower_Right"]
            event_info["quadrant"] = quadrant_names[event_idx]
            
            # 对于每个事件，记录其在页面上的位置信息（便于调试）
            if event_text:
                event_info["event_region"] = {
                    "min_x": min(item["x0"] for item in event_text),
                    "min_y": min(item["y0"] for item in event_text),
                    "max_x": max(item["x1"] for item in event_text),
                    "max_y": max(item["y1"] for item in event_text)
                }
            
            data.append(event_info)
    
    # 创建DataFrame并保存为TSV
    # 主要列
    main_columns = ["page", "event", "quadrant", "sample_name", "position", "oscillating_cn_2&3_states", "cn_segments"]
    # 其他列（可以用于调试）
    debug_columns = ["event_region", "full_text"]
    
    # 合并所有列
    all_columns = main_columns + debug_columns
    
    df = pd.DataFrame(data, columns=all_columns)
    
    # 保存完整数据（包括调试信息）
    df.to_csv(output_tsv + ".debug", sep='\t', index=False)
    
    # 保存正式数据（只包含主要列）
    df[main_columns].to_csv(output_tsv, sep='\t', index=False)
    
    print(f"已保存 {len(data)} 条染色体破裂事件信息到 {output_tsv}")
    print(f"调试信息已保存到 {output_tsv}.debug")

def extract_info_from_images(image_dir, output_tsv):
    """
    从已提取的图像中收集信息
    这个函数可以与从页面图像提取的事件信息结合
    """
    data = []
    pattern = r"page_(\d+)_part_(\d+)\.png"
    
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".png"):
            match = re.search(pattern, filename)
            if match:
                page_num = int(match.group(1))
                event_num = int(match.group(2))
                
                # 记录图像路径
                data.append({
                    "page": page_num,
                    "event": event_num,
                    "image_path": os.path.join(image_dir, filename)
                })
    
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_tsv, sep='\t', index=False)
        print(f"已从图像目录提取 {len(data)} 条记录")
    
    return data

def combine_text_and_image_info(text_tsv, image_tsv, output_tsv):
    """合并文本提取和图像提取的信息"""
    text_df = pd.read_csv(text_tsv, sep='\t')
    image_df = pd.read_csv(image_tsv, sep='\t')
    
    # 按页码和事件编号合并
    combined_df = pd.merge(
        text_df, 
        image_df, 
        on=["page", "event"], 
        how="outer"
    )
    
    combined_df.to_csv(output_tsv, sep='\t', index=False)
    print(f"已合并文本和图像信息，总共 {len(combined_df)} 条记录")

def main():
    # 设置路径
    pdf_path = "/Users/xurui/back_up_unit/天津大学文件/本科毕设相关/Article/ShatterSeek_data/NG_Supplementary_db_4.pdf"
    image_dir = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.PCAWG/SV_graph.dataset4"
    donor_ref_file = "/Users/xurui/back_up_unit/天津大学文件/本科毕设相关/Article/PCAWG-SupplementTable1.xlsx"

    # output path
    text_output = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.PCAWG/CNV_info_from_PDF/dataset4_info/extracted_events.tsv"
    image_output = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.PCAWG/CNV_info_from_PDF/dataset4_info/image_records.tsv"
    combined_output = "/Volumes/T7-shield/CS-Bachelor-Thesis/CNN_model/SV_graph.PCAWG/CNV_info_from_PDF/dataset4_info/combined_events.tsv"

    # 注意还需要在 get_fixed_quadrant_coordinates() 中设置正确的坐标范围
    
    # 确认路径
    print(f"PDF路径: {pdf_path}")
    print(f"参考文件路径: {donor_ref_file}")
    print(f"图像目录: {image_dir}")
    
    # 加载donor_ids参考
    donor_ids = load_donor_ids(donor_ref_file)
    
    # 显示固定坐标范围
    quadrants = get_fixed_quadrant_coordinates()
    quadrant_names = ["Upper_Left", "Upper_Right", "Lower_Left", "Lower_Right"]
    print("\n===== 固定区域坐标范围 =====")
    for i, quadrant in enumerate(quadrants):
        print(f"区域 {i+1} ({quadrant_names[i]}): X = [{quadrant['min_x']:.1f}, {quadrant['max_x']:.1f}], Y = [{quadrant['min_y']:.1f}, {quadrant['max_y']:.1f}]")
    
    # 测试部分：查看pages_text的结构和使用固定区域分组
    print("\n===== 测试：查看pages_text结构和使用固定区域分组 =====")
    # 只处理前几页进行测试
    test_pages = 4
    
    pages_text = extract_text_by_page(pdf_path)
    print(f"PDF总页数: {len(pages_text)}")
    
    # 只显示前test_pages页的信息
    for page_idx in range(min(test_pages, len(pages_text))):
        print(f"\n===== 第{page_idx+1}页 =====")
        page_text = pages_text[page_idx]
        print(f"文本块总数: {len(page_text)}")
        
        # 展示前几个文本块
        print("\n文本块示例:")
        for i in range(min(5, len(page_text))):
            text_item = page_text[i]
            print(f"块{i+1}: '{text_item['text']}' - 位置: ({text_item['x0']:.1f}, {text_item['y0']:.1f}) 到 ({text_item['x1']:.1f}, {text_item['y1']:.1f})")
        
        # 测试使用固定区域进行事件分组
        print("\n使用固定区域进行事件分组:")
        events = group_text_into_events(page_text)
        
        # 显示每个区域的文本块数量
        for event_idx, event_text in enumerate(events):
            print(f"区域 {event_idx+1} ({quadrant_names[event_idx]}): {len(event_text)}个文本块")
        
        # 详细查看每个区域的文本内容和提取的信息
        for event_idx, event_text in enumerate(events):
            if event_text:
                min_x = min(item["x0"] for item in event_text) if event_text else 0
                min_y = min(item["y0"] for item in event_text) if event_text else 0
                max_x = max(item["x1"] for item in event_text) if event_text else 0
                max_y = max(item["y1"] for item in event_text) if event_text else 0
                
                print(f"\n区域 {event_idx+1} ({quadrant_names[event_idx]}):")
                print(f"  实际内容区域: X = [{min_x:.1f}, {max_x:.1f}], Y = [{min_y:.1f}, {max_y:.1f}]")
                
                # 显示事件中的一些文本
                sample_texts = [item["text"] for item in event_text[:5]]
                print(f"  前几个文本: {sample_texts}")
                
                # 提取事件信息
                event_info = extract_event_info(event_text, donor_ids)
                print(f"  提取的信息:")
                print(f"    样本名称: {event_info['sample_name']}")
                print(f"    位置: {event_info['position']}")
                print(f"    震荡CN: {event_info['oscillating_cn_2&3_states']}")
                print(f"    CN片段数: {event_info['cn_segments']}")
                print()
                
    # 测试结束
    print("\n===== 测试结束 =====\n")
    
    # 询问用户是否继续处理所有页面
    response = input("是否继续处理所有页面并生成TSV文件? (y/n): ")
    if response.lower() == 'y':
        # 提取事件信息
        process_pdf_to_tsv(pdf_path, text_output, donor_ids)
        
        # 提取图像信息
        extract_info_from_images(image_dir, image_output)
        
        # 合并信息
        combine_text_and_image_info(text_output, image_output, combined_output)
        
        print("处理完成!")
    else:
        print("已取消处理")

if __name__ == "__main__":
    main() 