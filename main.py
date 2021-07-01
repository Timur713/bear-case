import cv2
import numpy as np

import utils

bears_count = 0


def read_image(filename):
    return cv2.imread(filename)


def get_original_filename(path):
    return path.split('/')[-1].split('.')[0]


def get_split_cords(image, by_x, by_y):
    y1 = np.linspace(0, len(list(image)), by_y).astype(int)
    x1 = np.linspace(0, len(list(image[0])), by_x).astype(int)
    return y1, x1


def bgr_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def hsv_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


def checkS(width, height):
    if 600 <= width * height <= 7600 and width <= 2.5 * height and height <= 2.5 * width:
        return True
    return False


filename = 'TEST IMAGES/withBears/22.JPG'
image = read_image(filename)

orig_name = get_original_filename(filename)

original_img = image.copy()

y1, x1 = get_split_cords(image, 7, 7)

image = bgr_to_hsv(image)

lower = np.array(utils.lower, dtype="uint8")
upper = np.array(utils.upper, dtype="uint8")


def find_contours(mask):
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0] if len(contours) == 2 else contours[1]


for i in range(1, y1.size):
    for j in range(1, x1.size):
        crop = image[y1[i - 1]:y1[i], x1[j - 1]:x1[j]]
        crop_copy = crop.copy()
        crop_copy = hsv_to_bgr(crop_copy)

        mask = cv2.inRange(crop, lower, upper)

        kernel = np.ones((11, 11), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours = find_contours(mask)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            if checkS(w, h):
                # draw crop contours
                cv2.rectangle(crop_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # increase contour size
                x -= 5
                y -= 5
                w += 5
                h += 5
                # draw original image contours
                cv2.rectangle(original_img, (x + x1[j - 1], y + y1[i - 1]), (x + x1[j - 1] + w, y + y1[i - 1] + h),
                              (0, 0, 255), 1)
                # increase bear count
                bears_count += 1
                # print found bear coordinates
                print(bears_count, "медведь : (", x + x1[j - 1], ',', y + y1[i - 1], ") площадь", w * h, '=', w, '*', h)

        cv2.imshow('mask', mask)
        cv2.imshow('original', crop_copy)
        cv2.waitKey()
cv2.destroyAllWindows()
result_filename = orig_name + "_{}_bears".format(bears_count) + ".jpg"
cv2.imwrite(result_filename, original_img)
res_img = read_image(result_filename)
res_img = cv2.resize(res_img, (1650, 900))
while (1):
    cv2.imshow(result_filename, res_img)
    k = cv2.waitKey(0) & 0xff
    # Exit if ESC pressed
    if k == 27:
        break
