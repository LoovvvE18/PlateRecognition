import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from process_utils import (
    img_read, to_gray, to_binary, gaussian_blur, canny_edge, invert_color,
    resize_image, morphology_opening, morphology_closing, add_weighted_subtract,
    otsu_threshold, hsv_conversion, color_mask_extraction, bitwise_and_with_mask,
    advanced_edge_processing, plate_recognition
)


class PlateRecognitionWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("车牌识别系统")
        self.window.geometry("1200x800")
        self.window.resizable(True, True)

        # 使窗口居中
        self.window.transient(parent)
        self.window.grab_set()

        # 存储图像数据
        self.original_image = None
        self.processed_image = None
        self.result_image = None
        self.current_file_path = None

        self.setup_ui()

    def setup_ui(self):
        """设置用户界面"""
        # 标题
        title_label = tk.Label(
            self.window,
            text="车牌识别系统",
            font=("微软雅黑", 20, "bold"),
            fg="#333",
            pady=20
        )
        title_label.pack()

        # 图片显示区域
        image_frame = tk.Frame(self.window)
        image_frame.pack(pady=20, fill=tk.BOTH, expand=True)

        # 第一个图片框 - 原始图片（可点击选择）
        left_frame = tk.Frame(image_frame)
        left_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        original_label = tk.Label(left_frame, text="选择图片", font=("微软雅黑", 12, "bold"))
        original_label.pack(pady=5)

        self.original_canvas = tk.Canvas(
            left_frame,
            width=350,
            height=250,
            bg="lightgray",
            relief=tk.SUNKEN,
            bd=2,
            cursor="hand2"
        )
        self.original_canvas.pack()

        self.original_canvas.create_text(
            175, 125,
            text="点击选择图片",
            font=("微软雅黑", 12),
            fill="gray"
        )

        self.original_canvas.bind("<Button-1>", self.select_image)

        # 第二个图片框 - 处理后图片
        middle_frame = tk.Frame(image_frame)
        middle_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        processed_label = tk.Label(middle_frame, text="车牌定位", font=("微软雅黑", 12, "bold"))
        processed_label.pack(pady=5)

        self.processed_canvas = tk.Canvas(
            middle_frame,
            width=350,
            height=250,
            bg="lightgray",
            relief=tk.SUNKEN,
            bd=2
        )
        self.processed_canvas.pack()

        self.processed_canvas.create_text(
            175, 125,
            text="车牌定位",
            font=("微软雅黑", 12),
            fill="gray"
        )

        # 第三个图片框 - 识别结果
        right_frame = tk.Frame(image_frame)
        right_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        result_label = tk.Label(right_frame, text="字符分割", font=("微软雅黑", 12, "bold"))
        result_label.pack(pady=5)

        self.result_canvas = tk.Canvas(
            right_frame,
            width=350,
            height=250,
            bg="lightgray",
            relief=tk.SUNKEN,
            bd=2
        )
        self.result_canvas.pack()

        self.result_canvas.create_text(
            175, 125,
            text="字符分割",
            font=("微软雅黑", 12),
            fill="gray"
        )

        # 文本框区域
        text_frame = tk.Frame(self.window)
        text_frame.pack(pady=20, fill=tk.X, padx=20)

        text_label = tk.Label(text_frame, text="识别信息", font=("微软雅黑", 12, "bold"))
        text_label.pack(anchor=tk.W)

        self.result_text = scrolledtext.ScrolledText(
            text_frame,
            height=8,
            font=("微软雅黑", 10),
            wrap=tk.WORD
        )
        self.result_text.pack(fill=tk.X, pady=5)

        # 按钮区域
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=20)

        # 开始识别按钮
        self.recognize_btn = tk.Button(
            button_frame,
            text="开始识别",
            width=20,
            height=2,
            font=("微软雅黑", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            command=self.start_recognition,
            state=tk.DISABLED  # 初始状态为禁用
        )
        self.recognize_btn.pack(side=tk.LEFT, padx=10)

        # 重置按钮
        reset_btn = tk.Button(
            button_frame,
            text="重置",
            width=15,
            height=2,
            font=("微软雅黑", 12),
            bg="#f44336",
            fg="white",
            command=self.reset_all
        )
        reset_btn.pack(side=tk.LEFT, padx=10)

        # 关闭按钮
        close_btn = tk.Button(
            button_frame,
            text="关闭",
            width=15,
            height=2,
            font=("微软雅黑", 12),
            bg="#9E9E9E",
            fg="white",
            command=self.window.destroy
        )
        close_btn.pack(side=tk.LEFT, padx=10)

    def select_image(self, event):
        """选择图片文件"""
        file_path = filedialog.askopenfilename(
            title="选择要识别的图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            try:
                self.current_file_path = file_path
                self.original_image = img_read(file_path)

                if self.original_image is None:
                    messagebox.showerror("错误", "无法读取图片文件")
                    return

                # 显示原始图片
                pil_image = self.cv2_to_pil(self.original_image)
                self.display_image(pil_image, self.original_canvas)

                # 启用识别按钮
                self.recognize_btn.config(state=tk.NORMAL)

                # 清空其他显示区域
                self.clear_canvas(self.processed_canvas, "预处理结果")
                self.clear_canvas(self.result_canvas, "识别结果")
                self.result_text.delete(1.0, tk.END)

                self.result_text.insert(tk.END, f"已选择图片: {os.path.basename(file_path)}\n")
                self.result_text.insert(tk.END,
                                        f"图片尺寸: {self.original_image.shape[1]} x {self.original_image.shape[0]}\n")
                self.result_text.insert(tk.END, "点击'开始识别'进行车牌识别...\n\n")

            except Exception as e:
                messagebox.showerror("错误", f"无法打开图片: {str(e)}")

    def cv2_to_pil(self, cv_image):
        """将OpenCV图像转换为PIL图像"""
        if len(cv_image.shape) == 3:
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(cv_image_rgb)
        else:
            return Image.fromarray(cv_image)

    def display_image(self, image, canvas):
        """在画布上显示图片"""
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1:
            canvas_width = 350
        if canvas_height <= 1:
            canvas_height = 250

        # 计算缩放比例
        img_width, img_height = image.size
        scale_x = (canvas_width - 20) / img_width
        scale_y = (canvas_height - 20) / img_height
        scale = min(scale_x, scale_y, 1.0)  # 不放大，只缩小

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(resized_image)

        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=photo,
            anchor=tk.CENTER
        )

        canvas.image = photo

    def clear_canvas(self, canvas, text):
        """清空画布并显示文本"""
        canvas.delete("all")
        canvas.create_text(
            175, 125,
            text=text,
            font=("微软雅黑", 12),
            fill="gray"
        )

    def start_recognition(self):
        """开始车牌识别"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先选择一张图片")
            return

        try:
            # 更新文本框
            self.result_text.insert(tk.END, "开始车牌识别处理...\n")
            self.window.update()

            # 第一步：预处理
            self.result_text.insert(tk.END, "步骤1: 图像预处理...\n")
            self.window.update()

            self.processed_image = plate_recognition(self.original_image)
            processed_pil = self.cv2_to_pil(self.processed_image)
            self.display_image(processed_pil, self.processed_canvas)

            self.result_text.insert(tk.END, "预处理完成\n")
            self.window.update()

            # 第二步：车牌检测和识别（这里你需要实现具体的识别逻辑）
            self.result_text.insert(tk.END, "步骤2: 车牌区域检测...\n")
            self.window.update()

            # 这里应该调用你的车牌检测和识别函数
            # 暂时用处理后的图像作为结果显示
            self.result_image = self.processed_image.copy()
            result_pil = self.cv2_to_pil(self.result_image)
            self.display_image(result_pil, self.result_canvas)

            self.result_text.insert(tk.END, "车牌检测完成\n")
            self.result_text.insert(tk.END, "步骤3: 字符识别...\n")
            self.window.update()

            # 模拟识别结果（你需要替换为实际的识别逻辑）
            recognized_text = "京A12345"  # 这里应该是实际识别的结果
            confidence = 0.95  # 识别置信度

            self.result_text.insert(tk.END, f"识别完成!\n")
            self.result_text.insert(tk.END, f"识别结果: {recognized_text}\n")
            self.result_text.insert(tk.END, f"置信度: {confidence:.2%}\n")
            self.result_text.insert(tk.END, "=" * 40 + "\n")

            messagebox.showinfo("完成", f"车牌识别完成!\n识别结果: {recognized_text}")

        except Exception as e:
            self.result_text.insert(tk.END, f"识别过程中出现错误: {str(e)}\n")
            messagebox.showerror("错误", f"车牌识别失败: {str(e)}")

    def reset_all(self):
        """重置所有内容"""
        self.original_image = None
        self.processed_image = None
        self.result_image = None
        self.current_file_path = None

        # 清空所有画布
        self.clear_canvas(self.original_canvas, "点击选择图片")
        self.clear_canvas(self.processed_canvas, "预处理结果")
        self.clear_canvas(self.result_canvas, "识别结果")

        # 清空文本框
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "系统已重置，请选择新的图片进行识别...\n")

        # 禁用识别按钮
        self.recognize_btn.config(state=tk.DISABLED)


class ImageProcessApp:
    def __init__(self, master):
        self.master = master
        master.title("数字图像处理与车牌识别系统")
        master.geometry("1000x900")  # 增加高度以容纳新的按钮行
        master.resizable(True, True)

        # 原图和处理后图片的变量
        self.original_image = None
        self.original_cv_image = None
        self.processed_image = None
        self.current_file_path = None

        # 大字标题
        title_label = tk.Label(
            master,
            text="数字图像处理与车牌识别系统",
            font=("微软雅黑", 22, "bold"),
            fg="#333",
            pady=20
        )
        title_label.pack()

        # 基础处理按钮框架
        basic_btn_frame = tk.Frame(master)
        basic_btn_frame.pack(pady=5)

        basic_label = tk.Label(basic_btn_frame, text="基础处理", font=("微软雅黑", 12, "bold"))
        basic_label.pack()

        basic_btn_container = tk.Frame(basic_btn_frame)
        basic_btn_container.pack(pady=5)

        basic_btns = [
            ("灰度化", self.gray_process),
            ("二值化", self.binary_process),
            ("高斯模糊", self.gaussian_process),
            ("Canny边缘", self.canny_process),
            ("色彩反转", self.invert_process)
        ]

        for text, cmd in basic_btns:
            btn = tk.Button(
                basic_btn_container,
                text=text,
                width=10,
                height=1,
                font=("微软雅黑", 10),
                command=cmd
            )
            btn.pack(side=tk.LEFT, padx=5)

        # 形态学处理按钮框架
        morph_btn_frame = tk.Frame(master)
        morph_btn_frame.pack(pady=5)

        morph_label = tk.Label(morph_btn_frame, text="形态学处理", font=("微软雅黑", 12, "bold"))
        morph_label.pack()

        morph_btn_container = tk.Frame(morph_btn_frame)
        morph_btn_container.pack(pady=5)

        morph_btns = [
            ("开运算", self.opening_process),
            ("闭运算", self.closing_process),
            ("加权减法", self.weighted_subtract_process),
            ("OTSU阈值", self.otsu_process)
        ]

        for text, cmd in morph_btns:
            btn = tk.Button(
                morph_btn_container,
                text=text,
                width=10,
                height=1,
                font=("微软雅黑", 10),
                command=cmd
            )
            btn.pack(side=tk.LEFT, padx=5)

        # 颜色处理按钮框架（移除了车牌识别按钮）
        color_btn_frame = tk.Frame(master)
        color_btn_frame.pack(pady=5)

        color_label = tk.Label(color_btn_frame, text="颜色处理", font=("微软雅黑", 12, "bold"))
        color_label.pack()

        color_btn_container = tk.Frame(color_btn_frame)
        color_btn_container.pack(pady=5)

        color_btns = [
            ("HSV转换", self.hsv_process),
            ("颜色掩码", self.color_mask_process),
            ("掩码应用", self.mask_apply_process),
            ("高级边缘", self.advanced_edge_process)
        ]

        for text, cmd in color_btns:
            btn = tk.Button(
                color_btn_container,
                text=text,
                width=10,
                height=1,
                font=("微软雅黑", 10),
                command=cmd
            )
            btn.pack(side=tk.LEFT, padx=5)

        # 车牌识别按钮框架（单独一行）
        plate_btn_frame = tk.Frame(master)
        plate_btn_frame.pack(pady=10)

        plate_label = tk.Label(plate_btn_frame, text="车牌识别", font=("微软雅黑", 12, "bold"))
        plate_label.pack()

        plate_btn_container = tk.Frame(plate_btn_frame)
        plate_btn_container.pack(pady=5)

        # 车牌识别按钮（更大尺寸，突出显示）
        plate_btn = tk.Button(
            plate_btn_container,
            text="车牌识别",
            width=15,
            height=2,
            font=("微软雅黑", 12, "bold"),
            bg="#4CAF50",  # 绿色背景
            fg="white",  # 白色文字
            command=self.plate_recognition_process
        )
        plate_btn.pack()

        # 图片显示区域
        image_frame = tk.Frame(master)
        image_frame.pack(pady=20, fill=tk.BOTH, expand=True)

        # 左侧原图显示框
        left_frame = tk.Frame(image_frame)
        left_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        original_label = tk.Label(left_frame, text="原图", font=("微软雅黑", 14, "bold"))
        original_label.pack(pady=5)

        self.original_canvas = tk.Canvas(
            left_frame,
            width=450,
            height=350,
            bg="lightgray",
            relief=tk.SUNKEN,
            bd=2,
            cursor="hand2"
        )
        self.original_canvas.pack()

        self.original_canvas.create_text(
            225, 175,
            text="点击选择图片",
            font=("微软雅黑", 12),
            fill="gray"
        )

        self.original_canvas.bind("<Button-1>", self.select_image)

        # 右侧处理后图片显示框
        right_frame = tk.Frame(image_frame)
        right_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)

        processed_label = tk.Label(right_frame, text="处理后", font=("微软雅黑", 14, "bold"))
        processed_label.pack(pady=5)

        self.processed_canvas = tk.Canvas(
            right_frame,
            width=450,
            height=350,
            bg="lightgray",
            relief=tk.SUNKEN,
            bd=2
        )
        self.processed_canvas.pack()

        self.processed_canvas.create_text(
            225, 175,
            text="处理后的图片将显示在这里",
            font=("微软雅黑", 12),
            fill="gray"
        )

    def select_image(self, event):
        """选择图片文件"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            try:
                self.current_file_path = file_path
                self.original_cv_image = img_read(file_path)

                if self.original_cv_image is None:
                    messagebox.showerror("错误", "无法读取图片文件")
                    return

                self.original_image = self.cv2_to_pil(self.original_cv_image)
                self.display_image(self.original_image, self.original_canvas)

                self.processed_canvas.delete("all")
                self.processed_canvas.create_text(
                    225, 175,
                    text="处理后的图片将显示在这里",
                    font=("微软雅黑", 12),
                    fill="gray"
                )

            except Exception as e:
                messagebox.showerror("错误", f"无法打开图片: {str(e)}")

    def cv2_to_pil(self, cv_image):
        """将OpenCV图像转换为PIL图像"""
        if len(cv_image.shape) == 3:
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(cv_image_rgb)
        else:
            return Image.fromarray(cv_image)

    def display_image(self, image, canvas):
        """在画布上显示图片"""
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1:
            canvas_width = 450
        if canvas_height <= 1:
            canvas_height = 350

        img_width, img_height = image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y)

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(resized_image)

        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=photo,
            anchor=tk.CENTER
        )

        canvas.image = photo

    def process_and_display(self, process_func, process_name):
        """通用的图像处理和显示函数"""
        if self.original_cv_image is None:
            messagebox.showwarning("警告", "请先选择一张图片")
            return

        try:
            processed_cv_image = process_func(self.original_cv_image)
            processed_pil_image = self.cv2_to_pil(processed_cv_image)
            self.display_image(processed_pil_image, self.processed_canvas)
            self.processed_image = processed_cv_image
            messagebox.showinfo("完成", f"{process_name}处理完成")

        except Exception as e:
            messagebox.showerror("错误", f"{process_name}处理失败: {str(e)}")

    # 基础处理函数
    def gray_process(self):
        self.process_and_display(to_gray, "灰度化")

    def binary_process(self):
        self.process_and_display(to_binary, "二值化")

    def gaussian_process(self):
        self.process_and_display(gaussian_blur, "高斯模糊")

    def canny_process(self):
        self.process_and_display(canny_edge, "Canny边缘检测")

    def invert_process(self):
        self.process_and_display(invert_color, "色彩反转")

    # 形态学处理函数
    def opening_process(self):
        self.process_and_display(morphology_opening, "形态学开运算")

    def closing_process(self):
        self.process_and_display(morphology_closing, "形态学闭运算")

    def weighted_subtract_process(self):
        self.process_and_display(add_weighted_subtract, "加权减法")

    def otsu_process(self):
        self.process_and_display(otsu_threshold, "OTSU自适应阈值")

    # 颜色处理函数
    def hsv_process(self):
        self.process_and_display(hsv_conversion, "HSV颜色空间转换")

    def color_mask_process(self):
        self.process_and_display(color_mask_extraction, "颜色掩码提取")

    def mask_apply_process(self):
        self.process_and_display(bitwise_and_with_mask, "掩码应用")

    def advanced_edge_process(self):
        self.process_and_display(advanced_edge_processing, "高级边缘处理")

    def plate_recognition_process(self):
        """打开车牌识别窗口"""
        PlateRecognitionWindow(self.master)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessApp(root)
    root.mainloop()