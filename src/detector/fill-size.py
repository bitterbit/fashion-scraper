import cv2
import numpy as np
import os.path
import sys

SIZE = (244, 244)

# fill image to be square
def fill_to_square(inputImage, fill):
    height, width, depth = inputImage.shape

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

    outputImage = cv2.copyMakeBorder(inputImage,
            top=m_top,
            bottom=m_bottom,
            left=m_left,
            right=m_right,
            borderType=cv2.BORDER_CONSTANT,
            value=fill)
    return outputImage


    cv2.imwrite('output.jpg', outputImage)
    cv2.imwrite('resized.jpg', resized)

def main():
    if len(sys.argv) < 3:
        print ("Usage: <src> <dst>")
        return

    src = sys.argv[1]
    dst = sys.argv[2]
    files = os.listdir(src)
    fill = [170, 170, 170] 

    for name in files:
        path = os.path.join(src, name)
        dst_path = os.path.join(dst, name)
        print(path, "->", dst_path)

        inputImage = cv2.imread(path, 1)
        if inputImage is not None:
            im = fill_to_square(inputImage, fill)
            resized = cv2.resize(im, SIZE, interpolation = cv2.INTER_AREA)
            cv2.imwrite(dst_path, resized)


if __name__ == '__main__':
    main()
