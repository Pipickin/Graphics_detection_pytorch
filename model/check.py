import cv2
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt


def get_image(path, reshape_size=(256, 256)):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, reshape_size)
    return img


def display_hist(img, eq, index):

    ax1 = plt.subplot(3, 2, index + 1)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    ax1.plot(cdf_normalized, color='b')
    ax1.hist(img.flatten(), 256, [0, 256], color='r')
    # ax1.lim([0, 256])
    ax1.legend(('cdf', 'histogram'), loc='upper left')
    # ax1.title(name)
    # plt.show()

    ax2 = plt.subplot(3, 2, index + 2)
    hist, bins = np.histogram(eq.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    ax2.plot(cdf_normalized, color='b')
    ax2.hist(eq.flatten(), 256, [0, 256], color='r')
    # ax2.lim([0, 256])
    ax2.legend(('cdf', 'histogram'), loc='upper left')



if __name__=='__main__':
    save_dir = './hist/'
    main_dir = '/media/shchetkov/HDD/media/images/task2/bask_source'
    paths = ['bask15.jpg', 'bask16.jpg', 'bask17.jpg']

    plt.subplot(321)
    i = 0
    res = []

    for img_path in paths:
        full_img_path = os.path.join(main_dir, img_path)
        image = get_image(full_img_path)
        equ = cv.equalizeHist(image)
        res.append(image)
        res.append(equ)
        # display_hist(image, equ, i)
        i += 2

    result = np.hstack(res)
    cv2.imwrite(os.path.join(save_dir, 'basket_ball.jpg'), result)
    cv2.imshow('result', result)
    plt.show()
    cv.waitKey(0)


