import cv2
import numpy as np

MAX_WIDTH = 1000  # 定义最大宽度常量


def img_read(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_binary(img):
    gray = to_gray(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)


def canny_edge(img):
    gray = to_gray(img)
    edges = cv2.Canny(gray, 100, 200)
    return edges


def invert_color(img):
    return cv2.bitwise_not(img)


def resize_image(img, max_width=MAX_WIDTH):
    """图像缩放"""
    pic_height, pic_width = img.shape[:2]
    if pic_width > max_width:
        resize_rate = max_width / pic_width
        img = cv2.resize(img, (max_width, int(pic_height * resize_rate)), interpolation=cv2.INTER_AREA)
    return img


def morphology_opening(img, kernel_size=(20, 20)):
    """形态学开运算"""
    if len(img.shape) == 3:  # 如果是彩色图像，先转为灰度
        img = to_gray(img)
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def morphology_closing(img, kernel_size=(4, 19)):
    """形态学闭运算"""
    if len(img.shape) == 3:  # 如果是彩色图像，先转为灰度
        img = to_gray(img)
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def add_weighted_subtract(img):
    """加权减法处理"""
    if len(img.shape) == 3:
        gray = to_gray(img)
    else:
        gray = img.copy()

    # 开运算
    kernel = np.ones((20, 20), np.uint8)
    img_opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # 加权减法
    result = cv2.addWeighted(gray, 1, img_opening, -1, 0)
    return result


def otsu_threshold(img):
    """OTSU自适应阈值"""
    if len(img.shape) == 3:
        gray = to_gray(img)
    else:
        gray = img.copy()

    # 先进行加权减法处理
    processed = add_weighted_subtract(gray)

    # OTSU阈值
    _, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def hsv_conversion(img):
    """BGR转HSV"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def color_mask_extraction(img):
    """颜色掩码提取（蓝、黄、绿色）"""
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义颜色范围
    lower_blue = np.array([100, 110, 110])
    upper_blue = np.array([130, 255, 255])
    lower_yellow = np.array([15, 55, 55])
    upper_yellow = np.array([50, 255, 255])
    lower_green = np.array([50, 50, 50])
    upper_green = np.array([100, 255, 255])

    # 创建掩码
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 合并掩码
    mask_sum = mask_blue + mask_yellow + mask_green
    return mask_sum


def bitwise_and_with_mask(img):
    """基于颜色掩码的按位与运算"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = color_mask_extraction(img)

    # 应用掩码
    result = cv2.bitwise_and(hsv, hsv, mask=mask)

    # 转换回BGR
    result_bgr = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    return result_bgr


def advanced_edge_processing(img):
    """高级边缘处理（Canny + 形态学操作）"""
    # 如果是彩色图像，先转为灰度
    if len(img.shape) == 3:
        gray = to_gray(img)
    else:
        gray = img.copy()

    # OTSU阈值处理
    thresh = otsu_threshold(gray)

    # Canny边缘检测
    edges = cv2.Canny(thresh, 100, 200)

    # 形态学闭运算
    kernel = np.ones((4, 19), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 形态学开运算
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return opened
