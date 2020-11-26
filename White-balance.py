import cv2
import numpy as np
import math
import matplotlib.pyplot as plt 
 
'''
    第一种简单的求均值白平衡法
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据'''
    
def mean_white_balance(img):
    # 读取图像
    b, g, r = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
    # 求各个通道所占增益
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img = cv2.merge([b, g, r])
    return balance_img



'''
完美反射白平衡
    STEP 1：计算每个像素的R\G\B之和
    STEP 2：按R+G+B值的大小计算出其前Ratio%的值作为参考点的的阈值T
    STEP 3：对图像中的每个点，计算其中R+G+B值大于T的所有点的R\G\B分量的累积和的平均值
    STEP 4：对每个点将像素量化到[0,255]之间
    依赖ratio值选取而且对亮度最大区域不是白色的图像效果不佳。
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
def perfect_reflective_white_balance(img_input):
    
    img = img_input.copy()
    b, g, r = cv2.split(img)
    m, n, t = img.shape
    sum_ = np.zeros(b.shape)
    # for i in range(m):
    #     for j in range(n):
    #         sum_[i][j] = int(b[i][j]) + int(g[i][j]) + int(r[i][j])
    sum_ = b.astype(np.int32) + g.astype(np.int32) + r.astype(np.int32)
 
    hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
    Y = 765
    num, key = 0, 0
    ratio = 0.01
    while Y >= 0:
        num += hists[Y]
        if num > m * n * ratio / 100:
            key = Y
            break
        Y = Y - 1
 
    # sum_b, sum_g, sum_r = 0, 0, 0
    # for i in range(m):
    #     for j in range(n):
    #         if sum_[i][j] >= key:
    #             sum_b += b[i][j]
    #             sum_g += g[i][j]
    #             sum_r += r[i][j]
    #             time = time + 1
    sum_b = b[sum_ >= key].sum()
    sum_g = g[sum_ >= key].sum()
    sum_r = r[sum_ >= key].sum()
    time = (sum_ >= key).sum()
 
    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time
 
    maxvalue = float(np.max(img))
    # maxvalue = 255
    # for i in range(m):
    #     for j in range(n):
    #         b = int(img[i][j][0]) * maxvalue / int(avg_b)
    #         g = int(img[i][j][1]) * maxvalue / int(avg_g)
    #         r = int(img[i][j][2]) * maxvalue / int(avg_r)
    #         if b > 255:
    #             b = 255
    #         if b < 0:
    #             b = 0
    #         if g > 255:
    #             g = 255
    #         if g < 0:
    #             g = 0
    #         if r > 255:
    #             r = 255
    #         if r < 0:
    #             r = 0
    #         img[i][j][0] = b
    #         img[i][j][1] = g
    #         img[i][j][2] = r
 
    b = img[:, :, 0].astype(np.int32) * maxvalue / int(avg_b)
    g = img[:, :, 1].astype(np.int32) * maxvalue / int(avg_g)
    r = img[:, :, 2].astype(np.int32) * maxvalue / int(avg_r)
    b[b > 255] = 255
    b[b < 0] = 0
    g[g > 255] = 255
    g[g < 0] = 0
    r[r > 255] = 255
    r[r < 0] = 0
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
 
    return img'''

a=cv2.imread("C:/s2.webp")
a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
image=mean_white_balance(a)
plt.subplot(121),plt.imshow(a)
plt.title("original image"),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(image)
plt.title("present image"),plt.xticks([]),plt.yticks([])
plt.show()
