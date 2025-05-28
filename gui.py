import cv2
import os
from process_utils import (
    img_read, to_gray, to_binary, gaussian_blur, canny_edge, invert_color,
    resize_image, morphology_opening, morphology_closing, add_weighted_subtract,
    otsu_threshold, hsv_conversion, color_mask_extraction, bitwise_and_with_mask,
    advanced_edge_processing
)
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
from plate_recognition import recognize_plate_from_image, img_read


class PlateRecognitionWindow:
    def __init__(self, parent=None):
        if parent is None:
            self.window = tk.Tk()
        else:
            self.window = tk.Toplevel(parent)
            self.window.transient(parent)
            self.window.grab_set()

        self.parent = parent
        self.window.title("车牌识别系统")
        self.window.geometry("1400x900")
        self.window.resizable(True, True)

        # 存储图像数据
        self.original_image = None
        self.current_file_path = None
        self.character_images = []

        self.setup_ui()
        self.center_window()

    def center_window(self):
        """将窗口居中显示"""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')

    def setup_ui(self):
        """设置用户界面"""
        # 主标题
        title_label = tk.Label(
            self.window,
            text="车牌识别系统",
            font=("微软雅黑", 20, "bold"),
            fg="#333"
        )
        title_label.pack(pady=(10, 20))

        # 创建主框架 - 左右分栏
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧面板 - 图像选择
        self.create_left_panel(main_frame)

        # 右侧面板 - 处理步骤和结果
        self.create_right_panel(main_frame)

        # 底部控制按钮
        #self.create_control_buttons()

    def create_left_panel(self, parent):
        """创建左侧面板 - 图像选择区域"""
        left_frame = tk.Frame(parent, relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_frame.pack_propagate(False)
        left_frame.configure(width=400, height=800)

        # 左侧标题
        left_title = tk.Label(
            left_frame,
            text="图像选择",
            font=("微软雅黑", 16, "bold"),
            fg="#2196F3"
        )
        left_title.pack(pady=(10, 20))

        # 原始图像显示区域
        image_label_frame = tk.LabelFrame(left_frame, text="原始图像", font=("微软雅黑", 12, "bold"))
        image_label_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.original_canvas = tk.Canvas(
            image_label_frame,
            bg="lightgray",
            relief=tk.SUNKEN,
            bd=2,
            cursor="hand2"
        )
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.original_canvas.create_text(
            180, 200,
            text="点击选择车牌图片\n\n支持格式:\nJPG, PNG, BMP, GIF",
            font=("微软雅黑", 14),
            fill="gray",
            justify=tk.CENTER
        )

        self.original_canvas.bind("<Button-1>", self.select_image)

        # # 图像信息显示
        # info_label_frame = tk.LabelFrame(left_frame, text="图像信息", font=("微软雅黑", 10, "bold"))
        # info_label_frame.pack(fill=tk.X, padx=10, pady=5)
        #
        # self.info_text = tk.Text(info_label_frame, height=6, font=("微软雅黑", 9), wrap=tk.WORD)
        # self.info_text.pack(fill=tk.X, padx=5, pady=5)

        # 识别结果预览
        result_preview_frame = tk.LabelFrame(left_frame, text="识别结果", font=("微软雅黑", 10, "bold"))
        result_preview_frame.pack(fill=tk.X, padx=10, pady=5)

        self.plate_number_label = tk.Label(
            result_preview_frame,
            text="车牌号码: 未识别",
            font=("微软雅黑", 16, "bold"),
            fg="red"
        )
        self.plate_number_label.pack(pady=5)

        self.plate_color_label = tk.Label(
            result_preview_frame,
            text="车牌颜色: 未识别",
            font=("微软雅黑", 12),
            fg="blue"
        )
        self.plate_color_label.pack(pady=2)

    def create_right_panel(self, parent):
        """创建右侧面板 - 处理步骤和结果"""
        right_frame = tk.Frame(parent, relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 右侧标题
        right_title = tk.Label(
            right_frame,
            text="处理步骤与结果",
            font=("微软雅黑", 16, "bold"),
            fg="#4CAF50"
        )
        right_title.pack(pady=(10, 15))

        # 创建主容器
        main_container = tk.Frame(right_frame)
        main_container.pack(fill=tk.X, padx=10, pady=5)

        # 1. 预处理步骤区域
        self.create_preprocessing_section(main_container)

        # 2. 颜色筛选区域
        self.create_color_filtering_section(main_container)

        # 3. 车牌提取和字符分割区域（同一行）
        self.create_plate_and_character_section(main_container)
        # 4. 控制按钮
        self.create_control_buttons(right_frame)



    def create_preprocessing_section(self, parent):
        """创建预处理步骤区域 - 只保留3个重要步骤"""
        preprocess_frame = tk.LabelFrame(
            parent,
            text="1. 图像预处理步骤",
            font=("微软雅黑", 12, "bold"),
            fg="#FF5722"
        )
        preprocess_frame.pack(fill=tk.X, pady=5)

        # 创建网格布局显示预处理步骤
        steps_container = tk.Frame(preprocess_frame)
        steps_container.pack(fill=tk.X, padx=5, pady=10)

        # 只保留3个重要预处理步骤
        self.preprocess_steps = [
            ("灰度转换", "gray"),
            ("二值化", "thresh"),
            ("边缘检测", "canny")
        ]

        self.preprocess_canvases = {}

        for i, (title, key) in enumerate(self.preprocess_steps):
            step_frame = tk.Frame(steps_container)
            step_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

            # 步骤标题
            step_label = tk.Label(step_frame, text=title, font=("微软雅黑", 11, "bold"))
            step_label.pack(pady=(0, 5))

            # 图像显示画布 - 统一大小
            canvas_widget = tk.Canvas(step_frame, width=180, height=120, bg="lightgray", relief=tk.SUNKEN, bd=1)
            canvas_widget.pack()

            canvas_widget.create_text(90, 60, text=title, font=("微软雅黑", 10), fill="gray")

            self.preprocess_canvases[key] = canvas_widget

    def create_color_filtering_section(self, parent):
        """创建颜色筛选区域"""
        color_frame = tk.LabelFrame(
            parent,
            text="2. 颜色筛选与车牌定位",
            font=("微软雅黑", 12, "bold"),
            fg="#9C27B0"
        )
        color_frame.pack(fill=tk.X, pady=5)

        color_container = tk.Frame(color_frame)
        color_container.pack(fill=tk.X, padx=5, pady=10)

        color_steps = [
            ("HSV转换", "hsv"),
            ("颜色掩码", "mask"),
            ("开运算", "color_filtered")
        ]

        self.color_canvases = {}

        for i, (title, key) in enumerate(color_steps):
            step_frame = tk.Frame(color_container)
            step_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

            step_label = tk.Label(step_frame, text=title, font=("微软雅黑", 11, "bold"))
            step_label.pack(pady=(0, 5))

            # 统一大小与预处理步骤一致
            canvas_widget = tk.Canvas(step_frame, width=180, height=120, bg="lightgray", relief=tk.SUNKEN, bd=1)
            canvas_widget.pack()

            canvas_widget.create_text(90, 60, text=title, font=("微软雅黑", 10), fill="gray")

            self.color_canvases[key] = canvas_widget

    def create_plate_and_character_section(self, parent):
        """创建车牌提取和字符分割区域（同一行）- 使用Grid布局确保等宽"""
        # 创建包含车牌提取和字符分割的容器
        plate_char_container = tk.Frame(parent)
        plate_char_container.pack(fill=tk.BOTH, expand=True, pady=5)

        # 配置网格权重，使两列等宽
        plate_char_container.grid_columnconfigure(0, weight=1)
        plate_char_container.grid_columnconfigure(1, weight=1)

        # 车牌提取区域（左侧）
        plate_frame = tk.LabelFrame(
            plate_char_container,
            text="3. 车牌区域提取",
            font=("微软雅黑", 12, "bold"),
            fg="#607D8B"
        )
        plate_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2.5))

        # 车牌显示
        self.plate_canvas = tk.Canvas(plate_frame, height=120, bg="lightgray", relief=tk.SUNKEN, bd=2)
        self.plate_canvas.pack(fill=tk.X, padx=10, pady=10)

        self.plate_canvas.create_text(200, 60, text="提取的车牌区域将显示在此", font=("微软雅黑", 10), fill="gray")

        # 字符分割区域（右侧）
        char_frame = tk.LabelFrame(
            plate_char_container,
            text="4. 字符分割与识别",
            font=("微软雅黑", 12, "bold"),
            fg="#795548"
        )
        char_frame.grid(row=0, column=1, sticky="nsew", padx=(2.5, 0))

        # 字符显示区域
        self.char_display_frame = tk.Frame(char_frame)
        self.char_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)

        # 默认提示
        default_label = tk.Label(
            self.char_display_frame,
            text="字符分割结果将显示在此处",
            font=("微软雅黑", 11),
            fg="gray"
        )
        default_label.pack()

    def create_control_buttons(self,parent):
        """创建控制按钮"""
        button_frame = tk.Frame(parent)
        button_frame.pack(pady=15)

        # 开始识别按钮
        self.recognize_btn = tk.Button(
            button_frame,
            text="开始识别",
            width=18,
            height=2,
            font=("微软雅黑", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            command=self.start_recognition,
            state=tk.DISABLED
        )
        self.recognize_btn.pack(side=tk.LEFT, padx=8)

        # # 重置按钮
        # reset_btn = tk.Button(
        #     button_frame,
        #     text="重置",
        #     width=15,
        #     height=2,
        #     font=("微软雅黑", 11),
        #     bg="#f44336",
        #     fg="white",
        #     command=self.reset_all
        # )
        # reset_btn.pack(side=tk.LEFT, padx=8)
        #
        # # 保存结果按钮
        # self.save_btn = tk.Button(
        #     button_frame,
        #     text="保存结果",
        #     width=15,
        #     height=2,
        #     font=("微软雅黑", 11),
        #     bg="#2196F3",
        #     fg="white",
        #     command=self.save_results,
        #     state=tk.DISABLED
        # )
        # self.save_btn.pack(side=tk.LEFT, padx=8)

        # 关闭按钮
        close_btn = tk.Button(
            button_frame,
            text="关闭",
            width=15,
            height=2,
            font=("微软雅黑", 11),
            bg="#9E9E9E",
            fg="white",
            command=self.close_window
        )
        close_btn.pack(side=tk.LEFT, padx=8)

    def select_image(self, event):
        """选择图片文件"""
        file_path = filedialog.askopenfilename(
            title="选择要识别的车牌图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("BMP", "*.bmp"),
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

            except Exception as e:
                messagebox.showerror("错误", f"无法打开图片: {str(e)}")

    def start_recognition(self):
        """开始车牌识别"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先选择一张图片")
            return

        try:
            # 调用车牌识别函数
            recognition_result = self.recognize_with_steps(self.original_image)

            if recognition_result['success']:
                self.display_all_steps(recognition_result)
                # 移除这行：self.save_btn.config(state=tk.NORMAL)

                plate_number = recognition_result['plate_number']
                plate_color = recognition_result['plate_color']

                # 更新左侧结果显示
                self.plate_number_label.config(
                    text=f"车牌号码: {plate_number}",
                    fg="green"
                )
                self.plate_color_label.config(text=f"车牌颜色: {plate_color}")

                messagebox.showinfo("完成", f"车牌识别完成!\n\n识别结果: {plate_number}\n车牌颜色: {plate_color}")

            else:
                error_msg = recognition_result.get('error', '未知错误')
                self.plate_number_label.config(text="车牌号码: 识别失败", fg="red")
                self.plate_color_label.config(text="车牌颜色: 未识别")
                messagebox.showerror("识别失败", f"车牌识别失败\n错误信息: {error_msg}")

        except Exception as e:
            messagebox.showerror("错误", f"车牌识别失败: {str(e)}")

    def recognize_with_steps(self, image):
        """执行车牌识别并返回详细步骤"""
        # 调用车牌识别函数
        result = recognize_plate_from_image(image)

        # 读取中间步骤图像 - 只读取需要的步骤
        step_files = {
            'gray': 'tmp/step3_gray.jpg',
            'thresh': 'tmp/step6_thresh.jpg',
            'canny': 'tmp/step7_canny.jpg',
            'hsv': 'tmp/step10_hsv.jpg',
            'mask': 'tmp/step13_back_to_bgr.jpg',
            'color_filtered': 'tmp/step16_color_open.jpg'
        }

        # 读取步骤图像
        for key, filename in step_files.items():
            if os.path.exists(filename):
                result[key] = cv2.imread(filename)

        # 读取字符分割图像
        char_images = []
        i = 0
        while True:
            for color in ['blue', 'yello', 'green']:
                char_file = f'tmp/plate_char_{i}_{color}.jpg'
                if os.path.exists(char_file):
                    char_images.append(cv2.imread(char_file))
                    break
            else:
                break
            i += 1

        result['character_images'] = char_images

        return result

    def display_all_steps(self, result):
        """显示所有处理步骤"""
        # 显示预处理步骤
        steps_mapping = {
            'gray': result.get('gray'),
            'thresh': result.get('thresh'),
            'canny': result.get('canny')
        }

        for key, image in steps_mapping.items():
            if image is not None and key in self.preprocess_canvases:
                pil_image = self.cv2_to_pil(image)
                if pil_image:
                    self.display_image(pil_image, self.preprocess_canvases[key], 250, 150)
                    self.window.update()

        # 显示颜色筛选步骤
        color_steps = {
            'hsv': result.get('hsv'),
            'mask': result.get('mask'),
            'color_filtered': result.get('color_filtered')
        }

        for key, image in color_steps.items():
            if image is not None and key in self.color_canvases:
                pil_image = self.cv2_to_pil(image)
                if pil_image:
                    self.display_image(pil_image, self.color_canvases[key], 250, 150)
                    self.window.update()

        # 显示车牌区域
        if result.get('roi') is not None:
            roi_pil = self.cv2_to_pil(result['roi'])
            self.display_plate_image(roi_pil, self.plate_canvas)
            self.window.update()

        # 显示字符分割结果
        self.display_characters(result.get('character_images', []), result.get('plate_number', ''))

    def display_plate_image(self, image, canvas):
        """专门用于显示车牌图像的方法 - 使用更大的缩放"""
        if image is None:
            return

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1:
            canvas_width = 400  # 增大默认宽度
        if canvas_height <= 1:
            canvas_height = 180  # 增大默认高度

        # 计算缩放比例 - 为车牌图像使用更大的缩放
        img_width, img_height = image.size
        scale_x = (canvas_width - 20) / img_width  # 减少边距以显示更大图像
        scale_y = (canvas_height - 20) / img_height
        scale = min(scale_x, scale_y, 2.0)  # 允许放大到2倍

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
    def display_characters(self, char_images, plate_number):
        """显示分割的字符"""
        # 清空之前的字符显示
        for widget in self.char_display_frame.winfo_children():
            widget.destroy()

        if not char_images:
            tk.Label(self.char_display_frame, text="未检测到字符",
                     font=("微软雅黑", 12), fg="red").pack()
            return

        # 创建字符显示容器
        chars_container = tk.Frame(self.char_display_frame)
        chars_container.pack(pady=5)

        # 标题
        title_label = tk.Label(chars_container, text="字符分割结果:", font=("微软雅黑", 10, "bold"))
        title_label.pack(pady=(0, 5))

        # 字符显示框架
        chars_frame = tk.Frame(chars_container)
        chars_frame.pack()

        # 显示每个字符
        for i, char_img in enumerate(char_images):
            char_frame = tk.Frame(chars_frame, relief=tk.RAISED, bd=1)
            char_frame.pack(side=tk.LEFT, padx=2)

            # 字符图像
            char_canvas = tk.Canvas(char_frame, width=40, height=60, bg="white")
            char_canvas.pack(padx=2, pady=2)

            if char_img is not None:
                char_pil = self.cv2_to_pil(char_img)
                if char_pil:
                    self.display_image(char_pil, char_canvas, 40, 60)

            # 字符标签
            char_text = plate_number[i] if i < len(plate_number) else '?'
            char_label = tk.Label(char_frame, text=char_text,
                                  font=("微软雅黑", 12, "bold"),
                                  fg="blue", bg="lightyellow")
            char_label.pack(pady=1)

    def display_image(self, image, canvas, max_width=None, max_height=None):
        """在画布上显示图片"""
        if image is None:
            return

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1:
            canvas_width = max_width or 200
        if canvas_height <= 1:
            canvas_height = max_height or 150

        # 计算缩放比例
        img_width, img_height = image.size
        scale_x = (canvas_width - 10) / img_width
        scale_y = (canvas_height - 10) / img_height
        scale = min(scale_x, scale_y, 1.0)

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

    def cv2_to_pil(self, cv_image):
        """将OpenCV图像转换为PIL图像"""
        if cv_image is None:
            return None
        if len(cv_image.shape) == 3:
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(cv_image_rgb)
        else:
            return Image.fromarray(cv_image)

    def save_results(self):
        """保存识别结果"""
        if not self.current_file_path:
            messagebox.showwarning("警告", "没有可保存的结果")
            return

        try:
            save_dir = filedialog.askdirectory(title="选择保存目录")
            if save_dir:
                base_name = os.path.splitext(os.path.basename(self.current_file_path))[0]

                # 保存识别报告
                report_file = os.path.join(save_dir, f"{base_name}_识别报告.txt")
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write("车牌识别结果报告\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"原始图片: {self.current_file_path}\n")
                    f.write(f"处理时间: 2025-05-28 06:23:40\n")
                    f.write(f"{self.plate_number_label.cget('text')}\n")
                    f.write(f"{self.plate_color_label.cget('text')}\n")

                messagebox.showinfo("保存成功", f"结果已保存到:\n{report_file}")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存结果时出错: {str(e)}")

    def reset_all(self):
        """重置所有内容 - 移除对已删除按钮的引用"""
        self.original_image = None
        self.current_file_path = None
        self.character_images = []

        # 重置显示
        self.reset_displays()

        # 重置左侧信息
        self.original_canvas.delete("all")
        self.original_canvas.create_text(
            200, 200,
            text="点击选择车牌图片\n\n支持格式:\nJPG, PNG, BMP, GIF",
            font=("微软雅黑", 14),
            fill="gray",
            justify=tk.CENTER
        )

        # 重置结果标签
        self.plate_number_label.config(text="车牌号码: 未识别", fg="red")
        self.plate_color_label.config(text="车牌颜色: 未识别")

        # 禁用识别按钮
        self.recognize_btn.config(state=tk.DISABLED)


    def reset_displays(self):
        """重置所有显示区域"""
        # 重置预处理显示
        for key, canvas in self.preprocess_canvases.items():
            canvas.delete("all")
            title = next(title for title, k in self.preprocess_steps if k == key)
            canvas.create_text(125, 75, text=title, font=("微软雅黑", 10), fill="gray")

        # 重置颜色筛选显示
        color_titles = {"hsv": "HSV转换", "mask": "颜色掩码", "color_filtered": "颜色筛选"}
        for key, canvas in self.color_canvases.items():
            canvas.delete("all")
            canvas.create_text(125, 75, text=color_titles.get(key, key), font=("微软雅黑", 10), fill="gray")

        # 重置车牌显示
        self.plate_canvas.delete("all")
        self.plate_canvas.create_text(200, 60, text="提取的车牌区域将显示在此", font=("微软雅黑", 10), fill="gray")

        # 重置车牌信息
        self.plate_info_text.delete(1.0, tk.END)

        # 重置字符显示
        for widget in self.char_display_frame.winfo_children():
            widget.destroy()

        default_label = tk.Label(
            self.char_display_frame,
            text="字符分割结果将显示在此处",
            font=("微软雅黑", 11),
            fg="gray"
        )
        default_label.pack()

    def close_window(self):
        """关闭窗口"""
        self.window.destroy()
        if self.parent is None:
            self.window.quit()

    def run(self):
        """运行GUI"""
        self.window.mainloop()


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