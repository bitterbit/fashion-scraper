import cv2
import numpy as np
import os.path
import sys 

DEFAULT_FILL = [170, 170, 170] 

def fill_to_square(input_img, fill=DEFAULT_FILL):
    height, width, depth = input_img.shape

    m_top = 0
    m_bottom = 0
    m_left = 0
    m_right = 0

    # height is not even, add 2 to width and 1 to height
    if height % 2 != 0:
        m_top = m_right = m_left = 1

    if width % 2 != 0:
        m_right = m_bottom = m_top = 1

    # we need more width
    if height > width:
        m_right += int((height - width) / 2)
        m_left  += int((height - width) / 2)

    # we need more height 
    if width > height:
        m_top    += int((width - height) / 2)
        m_bottom += int((width - height) / 2)

    outputImage = cv2.copyMakeBorder(input_img,
            top=m_top,
            bottom=m_bottom,
            left=m_left,
            right=m_right,
            borderType=cv2.BORDER_CONSTANT,
            value=fill)
    return outputImage

# will resize to a square image!
def resize(input_img, size):
    return cv2.resize(input_img, (size, size))
