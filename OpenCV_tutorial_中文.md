<h1 align = center >OpenCV3 python实践</h1>
<h4 align = right >update 2019.6.23</h4>

1. [requirement](#)
2. [图像阵列算数运算](#)
3. [图像Resize](#)
4. [图像亮度与对比](#)(待新增)
5. [图像色彩空间转换](#)
6. [图像像素统计](#)
7. [图像的均值及标准差](#) (待新增)
8. [图像归一化](#)
9. [ LUT查找表applyColormap](#)
10. [图像像素点索引](#)
11. [逻辑运算](#)
12. [知识点](#)
13. [滤波器](#)
    - 高通滤波(HBF)
    - 低通滤波(LPF)
14. [边缘检测](#)
15. [轮廓检测](#)
    - [方法](#)
16. [直线检测](#)
    - Hough 霍夫
17. [图像二值化](#)
18. [问题解决](#)

------

<h3 id=>requirement</h3>

1. Numpy 
2. Scipy
3. OpenNI (可选）
4. SensorKinect （可选）
5. Imutils

------

<h3 id=>图像阵列算数运算</h3>

下列依照顺序是加减乘除

```cv2.add(src1, src2, dst=None)``` 

```cv2.subtract(src1, src2, dst=None)``` 

```cv2.multiply(src1, src2, dst=None)```

```cv2.divide(src1, src2, dst=None)```

------

<h3 id=>图像Resize</h3>

```cv2.resize(src, (x, y), fx = None, fy = None)``` 

缩放方式可以有两种：

- 自定义像素缩放: 在x, y 的地方设定自定义的像素

- 按比例缩放：在fx, fy的地方 可以设定0.x, 假设输入0.5, 就按照原来的比例缩小50%， 如果都设为2， 就等于放大了一倍

------

<h3 id=>图像亮度与对比</h3>

### solution 1

公式 **dst = src1 * alpha+src2 * beta + gamma**

```cv2.addWeighted(src, alpha, np.zeros(img.shape, img.dtype), 0, beta)```

### Solution 2

```python
import cv2
import numpy as np


def increase_brightness(img, value=80):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
img = cv2.imread("spider-man2.jpg")
a = increase_brightness(img)
cv2.imshow("output", a)
cv2.waitKey()
cv2.destroyAllWindows()
```

### Solution 3

------

<h3 id=>图像色彩空间转换</h3>

```cv2.cvtColor(src, cv2.COLOR_效果)``` 

```cv2.inrange(src, low, hight)```:low值及high填入三个通道的范围ex.(65, 45, 192) 

在low以及high范围内的值， 会自动赋予255(白), 在范围外面的值会赋予0(黑)

------

<h3 id=>图像像素统计</h3>

```cv2.minMaxLoc(src)```: 输入灰度图or数组， 函数可以返回四个值 依序是：1.最小值 2.最大值 3最小值索引 4最大值索引

Example:

```python
import cv2 as cv
import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
cv.minMaxLoc(x)

>>>(1.0, 9.0, (0, 0), (2, 2))
```

------

<h3 id=>图像均值及标准差</h3>

```cv2.meanStdDev(src)``` : 该函数返回两个值 1.均值 2.标准差

------

<h3 id=>图像归一化</h3>

```cv2.normalize(src, dst, alpha, beta, norm_type=cv.归一化方法)```：

- NORM_MINMAX
- NORM_INF
- NORM_L1
- NORM_L2 最常用的就是NORM_MINMAX归一化方法。

------

<h3 id=>LUT查找表applyColormap</h3>

```cv2.applyColorMap(src, cv2.COLORMAP_效果)``` :将Colormap中封装的效果加到你的原图上



------

<h3 id=>图像像素点索引</h3>

```image[Row-min:max, Column-min:max, BGR[索引]]``` : 先行后列

**example:**

```
src1 = np.zeros(shape=[400, 400, 3]， dtype=np.uint8)
src1[100:200, 100:200, 0] = 255
```



<img src="/Users/stephenfang/Library/Application Support/typora-user-images/image-20190623165446543.png" style="zoom:50%"/>



------

<h3 id=>逻辑运算</h3>

```python
cv2.bitwise_and(src1, src2) :取重叠的像素A跟B都有的值
cv2.bitwise_xor(src1, src2) : A跟B各自的加上重叠后的值
cv2.bitwise_or(src1, src2): 重叠的像素不是A就是B
cv2.bitwise_not(src):not表示都不属于集合里面的元素，所有255-原来的值 = 反向后的值
```





------

<h3 id=>知识点</h3>

imwrite()函数要求图像为BGR or 灰度， 并且每个通道有一的bit
， 输出格式必须支持

例如 bmp要求通道有8位， png允许8 or 16位元

b = image[:,:,0]#得到蓝色通道

g = image[:,:,1]#得到绿色通道

r = image[:,:,2]#得到红色通道

------

<h3 id=>滤波器 - 高通滤波</h3>

检测图像某区域， 根据像素与周围像素的亮度来提升（boost)

<h3 id=>滤波器 - 低通滤波</h3>

<h4 id=>高斯模糊</h4>

cv2.GaussianBlur(src, (ksize), 0)

需要注意的是kernel size大小， 数值越大越模糊

------

<h3 id=>边缘检测滤波函数</h3>



<h4 id=>cv2.Canny</h4>

```python
cv2.Canny(src, dst, threshold1, threshold2, apertureSize, L2gradient)
```



src：輸入圖，單通道8位元圖。
dst：輸出圖，尺寸、型態和輸入圖相同。
threshold1：第一個閾值。
threshold2：第二個閾值。
apertureSize ：Sobel算子的核心大小。
L2gradient ：梯度大小的算法，預設為false。



边缘检测滤波函数

Laplacian()

sobel()

Scharr()

这些都会讲非边缘的区域转换成黑色
将边缘区域转换为白色

但是这些函数也容易将噪声错误地识别为边缘

解决办法是找到边缘之前对图像进行模糊处理

利用像是

blur()

medianBlur()

GasuussianBlur()

ksize 是滤波核（filter）的宽高， 奇数







------

<h3 id=>轮廓检测</h3>
<h4 id=>方法</h4>

```cv2.minAreaRect()``` :计算出包围目标的最小矩形区域

```cv2.boxpoint()``` : 计算出矩形顶点

Example :

```
rect = cv2.minAreaRect(c) #计算出包围目标的最小矩形区域
    box = cv2.boxPoints(rect) 
    box = np.int0(box) #浮点数转为整数
```

在图像上画出轮廓:

```cv2.drawContours(src, contours, contoursIdx, color, thickness=None, linetype=None) ```

ps.contoursIdx 表示的是轮廓数组的索引

Example:

```cv2.drawContours(img, [box], 0, (0, 0, 255), 3)```

其中的box是上面所求出来的轮廓数组， 必须是列表形式

<br>
<br>
<br>

```cv2.minEnclosingCircle()```: 返回 1.包围目标的元中心坐标 2. 圆的半径radius

Example:

```python:
(x, y), radius = cv2.minEnclosingCircle(c)
    center = (int(x), int(y)) #整数
    radius = int(radius)
    img3 = cv2.circle(img3, center, radius, (0, 255, 0), 2) #在图像上画出来
```



------

<h3 id=> 直线检测 - Hough</h3>

1. HoughLines
2. HoughLinesP(img, rho, theta, minLineLength, maxLineGap)

- dst:    输出图像. 它应该是个灰度图 (但事实上是个二值化图) 
- rho :   参数极径 r 以像素值为单位的分辨率. 我们使用 1 像素.
- theta:  参数极角 \theta 以弧度为单位的分辨率. 我们使用 1度 (即CV_PI/180)
- threshold:    设置阈值： 一条直线所需最少的的曲线交点。超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少。
- minLinLength: 能组成一条直线的最少点的数量. 点数量不足的直线将被抛弃.

- maxLineGap:   能被认为在一条直线上的两点的最大距离。



Example：



```python
img = cv2.imread("lane.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(gray, 50, 120)

minLineLength = 20
maxLineGap = 5
lines = cv2.HoughLinesP(edge, 
                        1, np.pi/180, 
                        100, 
                        minLineLength, maxLineGap)

for x1, y1, x2, y2 in lines[0]:
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("edge", edge)
cv2.imshow("lines", img)
cv2.waitKey()
cv2.destroyAllWindows()
```



<h3 id=> 圓检测 - Hough</h3>

用到的是HoughCircles 這個方法

跟直線類似，有一個圓心間的最小距離和圓的最小及最大半徑

```python
HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None)
```

- image :8-bit, single-channel, grayscale input image.  image输入必须是8位的单通道灰度图像
- method: 目前只有HOUGH_GRADIENT,也就是2-1霍夫变换
- dp: 原图像和累加器juzh矩阵的像素比 一般设为1就可以了
- minDist: 圆心center中圆心之间的最小圆心距 如果小于此值,则认为两个是同一个圆(此时抛弃该圆心点,防止圆过于重合)
- circles: Output vector of found circles.Each vector is encoded as 3 or 4 element circle也就是我们最后圆的结果集
- param1 canny双阈值边缘检测的高阈值,经查阅一般低阈值位高阈值的0.4或者0.5
- param2 在确定圆心时 圆周点的梯度的累加器投票数ddata以及在确定半径时相同圆心相同半径的圆的个数max_count必须大于此阈值才认为是合法的圆心和半径
- minRadius 最小的半径 如果过小可能会因为噪音导致找到许多杂乱无章的圆,过大会找不到圆
- minRadius 最大的半径 如果过小可能会找不到圆,过大会找很多杂乱无章的圆



------

<h3 id=>图像二值化-非黑即白</h3>

使用cv2.threshold(src, 分类阈值, 不合格所赋予的值， 方法) 

Example:

```
img = cv2.imread("lane.jpg", 0)
ret, thresh = cv2.threshold(img, 127, 255, 0)
cv2.imshow("wow", thresh)
cv2.waitKey()
cv2.destroyAllWindows()
```

------

<h3 align = center> 问题解决 </h3>

1. 问题一

FindContours support only 8uC1 and 32sC1 images in function cvStartFindContours

只支持 single channel, unit8

1. 解决
   输入findContours函数之前， 先将图像进行转换成unit8, channel = 1的

```cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)```
