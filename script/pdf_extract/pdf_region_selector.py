# 先使用pdf_region_selector.py选择区域，然后使用pdf_event_extractor.py提取PDF中的文字部分 [检查]
# 按s保存后，输出文件为与PDF同名的.py文件，作为pdf_event_extractor.py的输入

import fitz  # PyMuPDF
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import os

class PDFRegionSelector:
    def __init__(self):
        self.pdf_path = None
        self.current_page = 0
        self.total_pages = 0
        self.zoom = 2.0  # 缩放因子，使显示更清晰
        self.regions = {}  # 存储每页的区域
        self.current_regions = []  # 当前页面的区域
        self.start_point = None
        self.end_point = None
        self.dragging = False
        self.region_count = 0  # 当前页面的区域计数
        
    def select_pdf(self):
        """选择PDF文件"""
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        self.pdf_path = filedialog.askopenfilename(
            title="选择PDF文件",
            filetypes=[("PDF文件", "*.pdf")]
        )
        root.destroy()
        
        if not self.pdf_path:
            print("未选择任何文件")
            return False
            
        print(f"已选择文件: {self.pdf_path}")
        return True
        
    def load_pdf(self):
        """加载PDF文件"""
        if not self.pdf_path:
            return False
            
        try:
            self.doc = fitz.open(self.pdf_path)
            self.total_pages = len(self.doc)
            print(f"PDF共有 {self.total_pages} 页")
            return True
        except Exception as e:
            print(f"加载PDF出错: {e}")
            return False
            
    def render_page(self):
        """渲染当前页并显示"""
        if not hasattr(self, 'doc'):
            print("未加载PDF文件")
            return
            
        page = self.doc[self.current_page]
        pix = page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom))
        
        # 转换为OpenCV图像
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        if pix.n == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        self.original_img = img.copy()
        self.display_img = img.copy()
        
        # 显示已经选择的区域
        self.region_count = len(self.current_regions) if self.current_page in self.regions else 0
        self.current_regions = self.regions.get(self.current_page, [])
        
        for i, (start, end) in enumerate(self.current_regions):
            cv2.rectangle(self.display_img, start, end, (0, 255, 0), 2)
            cv2.putText(self.display_img, f"区域 {i+1}", (start[0], start[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # 显示页码和帮助信息
        cv2.putText(self.display_img, f"第 {self.current_page+1}/{self.total_pages} 页", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(self.display_img, "左键拖动: 选择区域", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(self.display_img, "a/d: 上一页/下一页", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(self.display_img, "c: 清除当前页所有区域", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(self.display_img, "s: 保存所有区域到文件", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(self.display_img, "q: 退出", (10, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        cv2.imshow("PDF区域选择工具", self.display_img)
            
    def mouse_callback(self, event, x, y, flags, param):
        """处理鼠标事件"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.dragging = True
            
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            # 更新显示临时框
            img_copy = self.display_img.copy()
            cv2.rectangle(img_copy, self.start_point, (x, y), (0, 255, 255), 2)
            cv2.imshow("PDF区域选择工具", img_copy)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.dragging = False
            
            # 确保起点和终点正确（左上到右下）
            x1, y1 = min(self.start_point[0], self.end_point[0]), min(self.start_point[1], self.end_point[1])
            x2, y2 = max(self.start_point[0], self.end_point[0]), max(self.start_point[1], self.end_point[1])
            self.start_point = (x1, y1)
            self.end_point = (x2, y2)
            
            # 添加到区域列表
            if self.current_page not in self.regions:
                self.regions[self.current_page] = []
                
            self.regions[self.current_page].append((self.start_point, self.end_point))
            self.current_regions = self.regions[self.current_page]
            self.region_count = len(self.current_regions)
            
            # 重新渲染页面以显示新区域
            self.render_page()
            
    def convert_to_pdf_coords(self, regions):
        """将屏幕坐标转换为PDF坐标"""
        converted_regions = {}
        
        for page_num, page_regions in regions.items():
            page = self.doc[page_num]
            page_width = page.rect.width
            page_height = page.rect.height
            
            converted_page_regions = []
            for (start_x, start_y), (end_x, end_y) in page_regions:
                # 将缩放的屏幕坐标转换回PDF坐标
                pdf_start_x = start_x / self.zoom
                pdf_start_y = start_y / self.zoom
                pdf_end_x = end_x / self.zoom
                pdf_end_y = end_y / self.zoom
                
                converted_page_regions.append({
                    'top_left': (pdf_start_x, pdf_start_y),
                    'bottom_right': (pdf_end_x, pdf_end_y),
                    'width': pdf_end_x - pdf_start_x,
                    'height': pdf_end_y - pdf_start_y
                })
                
            converted_regions[page_num + 1] = converted_page_regions  # 页码从1开始
            
        return converted_regions
        
    def save_regions(self):
        """保存区域信息到文件"""
        if not self.regions:
            print("没有选择任何区域")
            return
            
        # 转换坐标
        pdf_regions = self.convert_to_pdf_coords(self.regions)
        
        # 准备输出文件名
        pdf_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
        output_file = f"{pdf_name}_regions.py"
        
        with open(output_file, 'w') as f:
            f.write("# PDF区域信息\n\n")
            f.write(f"pdf_path = \"{self.pdf_path}\"\n\n")
            f.write("# 格式: 页码: [{'top_left': (x1, y1), 'bottom_right': (x2, y2), 'width': w, 'height': h}, ...]\n")
            f.write("# 坐标单位为PDF点 (point)\n")
            f.write("regions = {\n")
            
            for page_num, regions in sorted(pdf_regions.items()):
                f.write(f"    {page_num}: [\n")
                for region in regions:
                    f.write(f"        {{\n")
                    f.write(f"            'top_left': {region['top_left']},\n")
                    f.write(f"            'bottom_right': {region['bottom_right']},\n")
                    f.write(f"            'width': {region['width']},\n")
                    f.write(f"            'height': {region['height']}\n")
                    f.write(f"        }},\n")
                f.write(f"    ],\n")
            
            f.write("}\n")
        
        print(f"区域信息已保存到 {output_file}")
        
    def run(self):
        """运行区域选择工具"""
        if not self.select_pdf():
            return
            
        if not self.load_pdf():
            return
            
        cv2.namedWindow("PDF区域选择工具")
        cv2.setMouseCallback("PDF区域选择工具", self.mouse_callback)
        
        self.render_page()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # 退出
                break
                
            elif key == ord('a'):  # 上一页
                if self.current_page > 0:
                    self.current_page -= 1
                    self.render_page()
                    
            elif key == ord('d'):  # 下一页
                if self.current_page < self.total_pages - 1:
                    self.current_page += 1
                    self.render_page()
                    
            elif key == ord('c'):  # 清除当前页区域
                if self.current_page in self.regions:
                    self.regions[self.current_page] = []
                    self.current_regions = []
                    self.region_count = 0
                    self.render_page()
                    
            elif key == ord('s'):  # 保存区域信息
                self.save_regions()
                
        cv2.destroyAllWindows()
        self.doc.close()

if __name__ == "__main__":
    selector = PDFRegionSelector()
    selector.run() 