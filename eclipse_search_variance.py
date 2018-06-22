import itertools
import operator as op
import numpy as np
import cv2
from PIL import Image
#ビット周辺ごとに閾値の微調整を行う必要性有りか
#分散でビットをみた
#ビット周辺のうち標準偏差が閾値以上のものを
#

class esbit:
    def __init__(self, num_of_fan, num_of_section):
        self.non_zero = np.zeros((num_of_fan, num_of_section), dtype=np.int64)
        self.mean_all = np.float64(0.)
        self.deviations = np.zeros(num_of_fan, dtype=np.float64)
        self.singular = np.int64(0) # singular direction
        self.variance = np.float64(0.)
        self.direction = np.zeros(2, dtype=np.float64)



def to_mono(img, t=127):
    edge = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    return cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)[1]

#ふちに関してはspan/2　回して大きな偏差が取れる方を保存
def scan_type_f360(src_mono, y, x, radius, num_of_split_degree, num_of_split_radius=1):
    center=(x ,y)
    color=255
    thickness = -1
    outer_end_idx = np.trunc(num_of_split_radius / 2) - 1
    inner_start_idx = outer_end_idx if num_of_split_radius == (outer_end_idx+1)*2 else outer_end_idx+1
    
    degree_fan = np.ceil(360 / num_of_split_degree)
    span_degree_center_lline = round(360 / num_of_split_degree)
    span_radius = round( radius / num_of_split_radius )
    
    result = esbit(num_of_split_degree, num_of_split_radius)

    for fan_number in range(num_of_split_degree):
        degree_center_lline = fan_number * span_degree_center_lline
        tmp_result = np.zeros(num_of_split_radius)
        masked_img = src_mono
        
        angle = 0
        startAngle = ( degree_center_lline - ( degree_fan / 2 ) )
        endAngle = ( degree_center_lline + ( degree_fan / 2 ) )

        for j, span_radius_iter in enumerate(range(radius, span_radius-1, -span_radius)):
            mask_current = cv2.ellipse(
                cv2.threshold(np.zeros_like(src_mono), 200, 255, cv2.THRESH_BINARY)[1], 
                center, 
                (span_radius_iter,)*2, 
                angle, 
                startAngle, 
                endAngle, 
                color, 
                thickness
            )
            masked_img = cv2.bitwise_and(masked_img, masked_img, mask=mask_current)
            tmp_result[j] = cv2.countNonZero(masked_img)

        #if tmp_result[:outer_end_idx].sum() < tmp_result[inner_start_idx:].sum():
        result.non_zero[fan_number] = np.append(np.diff(tmp_result, axis=-1), tmp_result[-1])
        #else:
        #    result[fan_number] = np.nan
    result.mean_all = np.mean(result.non_zero)
    result.deviations = [non_zero_a_fan.sum() - result.mean_all for non_zero_a_fan in result.non_zero]
    result.singular = np.argmax(result.deviations)# * span_degree_center_lline
    result.direction[0] = np.cos(result.singular)
    result.direction[1] = np.sin(result.singular)

    return result


def none(filename, radius, degree_split, radius_split, not_noise_ikiti):
    mono = to_mono(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
    new_img = np.zeros_like(mono, dtype=np.int8)

    print(mono)
    print(cv2.countNonZero(mono))
    for y, line_x in enumerate(mono):
        for x, dot in enumerate(line_x):
            if 255 <= dot:
                bit_dist = scan_type_f360(mono, y, x, radius, degree_split).non_zero
                variance = np.var(bit_dist)
                print(variance)
                if not_noise_ikiti < variance:
                    new_img[y][x] = 255

    Image.fromarray(new_img , 'L').show()
    Image.fromarray(new_img , 'L').save('lenna_changed.png')

iki = 25


none("s0_edge_img_mono.jpg", 15, 12, 1, iki)

