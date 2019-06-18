<h1 align = center >OpenCV3 python实践</h1>
<h4 align = right >update 2019.6.16</h4>

1. [requirement](#)
2. [知识点](#)
3. [滤波器](#)
	- 高通滤波(HBF)
	- 低通滤波(LPF)
4. [边缘检测](#)
1. [轮廓检测](#)
	- [方法](#)
1. [直线检测](#)
	- Hough 霍夫
5. [图像二值化](#)
6. [问题解决](#)

---

<h3 id=>requirement</h3>

1. Numpy 
2. Scipy
3. OpenNI (可选）
4. SensorKinect （可选）

---

<h3 id=>知识点</h3>

imwrite()函数要求图像为BGR or 灰度， 并且每个通道有一的bit
， 输出格式必须支持

例如 bmp要求通道有8位， png允许8 or 16位元


b = image[:,:,0]#得到蓝色通道

g = image[:,:,1]#得到绿色通道

r = image[:,:,2]#得到红色通道

---
<h3 id=>滤波器 - 高通滤波</h3>

检测图像某区域， 根据像素与周围像素的亮度来提升（boost)

---


<h3 id=>边缘检测滤波函数</h3>

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

---

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




---

<h3 id=> 直线检测 </h3>

<h4 id=> Hough 霍夫 </h4>

1. HoughLines
2. HoughLinesP

Example：



```
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

---

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


---

<h3 align = center> 问题解决 </h3>

1. 问题一

FindContours support only 8uC1 and 32sC1 images in function cvStartFindContours

只支持 single channel, unit8

1. 解决
输入findContours函数之前， 先将图像进行转换成unit8, channel = 1的

```cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)```
