import os
import cv2
import numpy as np
import img_math
import img_recognition

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


class CardPredictor:
    def __init__(self):
        self.model = None
        self.modelchinese = None

    def train_svm(self):
        self.model = SVM(C=1, gamma=0.5)
        self.modelchinese = SVM(C=1, gamma=0.5)
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        if os.path.exists("svmchinese.dat"):
            self.modelchinese.load("svmchinese.dat")

    def img_first_pre(self, car_pic_file):
        if type(car_pic_file) == type(""):
            img = cv2.imread(car_pic_file)
        else:
            img = car_pic_file
        pic_hight, pic_width = img.shape[:2]
        if pic_width > MAX_WIDTH:
            resize_rate = MAX_WIDTH / pic_width
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        cv2.imwrite("tmp/step1_resize.jpg", img)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imwrite("tmp/step2_gaussian.jpg", img)
        oldimg = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("tmp/step3_gray.jpg", img)
        Matrix = np.ones((20, 20), np.uint8)
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, Matrix)
        cv2.imwrite("tmp/step4_opening.jpg", img_opening)
        img_opening_weighted = cv2.addWeighted(img, 1, img_opening, -1, 0)
        cv2.imwrite("tmp/step5_addWeighted.jpg", img_opening_weighted)
        ret, img_thresh = cv2.threshold(img_opening_weighted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite("tmp/step6_thresh.jpg", img_thresh)
        img_edge = cv2.Canny(img_thresh, 100, 200)
        cv2.imwrite("tmp/step7_canny.jpg", img_edge)
        Matrix = np.ones((4, 19), np.uint8)
        img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, Matrix)
        cv2.imwrite("tmp/step8_close.jpg", img_edge1)
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
        cv2.imwrite("tmp/step9_open_again.jpg", img_edge2)
        return img_edge2, oldimg

    def img_only_color(self, filename, oldimg, img_contours):
        pic_hight, pic_width = img_contours.shape[:2]
        lower_blue = np.array([100, 110, 110])
        upper_blue = np.array([130, 255, 255])
        lower_yellow = np.array([15, 55, 55])
        upper_yellow = np.array([50, 255, 255])
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([100, 255, 255])
        hsv = cv2.cvtColor(filename, cv2.COLOR_BGR2HSV)
        cv2.imwrite("tmp/step10_hsv.jpg", hsv)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_yellow, upper_green)
        mask_sum = mask_blue + mask_yellow + mask_green
        cv2.imwrite("tmp/step11_mask.jpg", mask_sum)
        output = cv2.bitwise_and(hsv, hsv, mask=mask_sum)
        cv2.imwrite("tmp/step12_bitwise_and_hsv.jpg", output)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
        cv2.imwrite("tmp/step13_back_to_bgr.jpg", output_bgr)
        output_gray = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("tmp/step14_gray_from_bgr.jpg", output_gray)
        Matrix = np.ones((20, 20), np.uint8)
        img_edge1 = cv2.morphologyEx(output_gray, cv2.MORPH_CLOSE, Matrix)
        cv2.imwrite("tmp/step15_color_close.jpg", img_edge1)
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
        cv2.imwrite("tmp/step16_color_open.jpg", img_edge2)
        card_contours = img_math.img_findContours(img_edge2)
        card_imgs = img_math.img_Transform(card_contours, oldimg, pic_width, pic_hight)
        colors, car_imgs = img_math.img_color(card_imgs)
        predict_result = []
        predict_str = ""
        roi = None
        card_color = None
        for i, color in enumerate(colors):
            if color in ("blue", "yello", "green"):
                card_img = card_imgs[i]
                cv2.imwrite(f"tmp/card_plate_{color}.jpg", card_img)
                try:
                    gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                except:
                    continue
                if color == "green" or color == "yello":
                    gray_img = cv2.bitwise_not(gray_img)
                ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imwrite(f"tmp/card_plate_thresh_{color}.jpg", gray_img)
                x_histogram = np.sum(gray_img, axis=1)
                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2
                wave_peaks = img_math.find_waves(x_threshold, x_histogram)
                if len(wave_peaks) == 0:
                    continue
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                gray_img = gray_img[wave[0]:wave[1]]
                row_num, col_num = gray_img.shape[:2]
                gray_img = gray_img[1:row_num - 1]
                y_histogram = np.sum(gray_img, axis=0)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5
                wave_peaks = img_math.find_waves(y_threshold, y_histogram)
                if len(wave_peaks) < 6:
                    continue
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)
                cur_dis = 0
                for j, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if j > 0:
                    wave = (wave_peaks[0][0], wave_peaks[j][1])
                    wave_peaks = wave_peaks[j + 1:]
                    wave_peaks.insert(0, wave)
                point = wave_peaks[2]
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)
                if len(wave_peaks) <= 6:
                    continue
                part_cards = img_math.seperate_card(gray_img, wave_peaks)
                for k, part_card in enumerate(part_cards):
                    if np.mean(part_card) < 255 / 5:
                        continue
                    part_card_old = part_card
                    w = abs(part_card.shape[1] - SZ) // 2
                    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(f"tmp/plate_char_{k}_{color}.jpg", part_card)
                    part_card = img_recognition.preprocess_hog([part_card])
                    if k == 0:
                        resp = self.modelchinese.predict(part_card)
                        charactor = img_recognition.provinces[int(resp[0]) - PROVINCE_START]
                    else:
                        resp = self.model.predict(part_card)
                        charactor = chr(int(resp[0]))
                    if charactor == "1" and k == len(part_cards) - 1:
                        if part_card_old.shape[0] / part_card_old.shape[1] >= 7:
                            continue
                    predict_result.append(charactor)
                    predict_str = "".join(predict_result)
                roi = card_img
                card_color = color
                break
        return predict_str, roi, card_color


# 添加便于GUI调用的函数
def recognize_plate_from_image(image):
    """
    从图像中识别车牌

    参数:
        image: cv2图像对象或图像文件路径

    返回:
        dict: 包含识别结果的字典
        {
            'plate_number': str,     # 识别的车牌号
            'plate_color': str,      # 车牌颜色
            'roi': numpy.ndarray,    # 提取的车牌区域图像
            'edge_image': numpy.ndarray,  # 边缘检测结果图像
            'success': bool          # 是否识别成功
        }
    """
    try:
        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        predictor = CardPredictor()
        predictor.train_svm()

        # 第一步：预处理
        img_edge2, oldimg = predictor.img_first_pre(image)

        # 第二步：车牌识别
        if type(image) == type(""):
            img_bgr = cv2.imread(image)
        else:
            img_bgr = image

        result, roi, color = predictor.img_only_color(img_bgr, oldimg, img_edge2)

        return {
            'plate_number': result if result else "",
            'plate_color': color if color else "",
            'roi': roi,
            'edge_image': img_edge2,
            'success': bool(result and roi is not None)
        }
    except Exception as e:
        print(f"车牌识别错误: {str(e)}")
        return {
            'plate_number': "",
            'plate_color': "",
            'roi': None,
            'edge_image': None,
            'success': False,
            'error': str(e)
        }


def img_read(filename):
    """读取图片文件"""
    return cv2.imread(filename)


if __name__ == "__main__":
    # 这里替换为你的图片路径
    image_path = "1.jpg"
    result = recognize_plate_from_image(image_path)
    print("识别结果:", result['plate_number'])
    print("车牌颜色:", result['plate_color'])
    print("识别成功:", result['success'])
    print("每一步处理图片已保存在 tmp/ 目录下")