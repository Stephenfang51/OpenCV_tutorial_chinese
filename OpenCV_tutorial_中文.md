<h1 align = center >OpenCV3 python实践</h1>
<h4 align = right >update 2019.8.6</h4>

1. [requirement](#)

2. [图像阵列算数运算](#)

3. [BGR/RGB 颜色互转](#)

4. [图像Resize与插值法](#)

5. [图像亮度与对比](#)(待新增)

6. [图像色彩空间转换](#)

7. [图像像素统计](#)

8. [图像的均值及标准差](#) (待新增)

9. [图像归一化](#)

10. [ LUT查找表applyColormap](#)

11. [图像像素点索引](#)

12. [逻辑运算](#)

13. [ROI区域操作](#)

14. [图像直方图统计/均衡化](#)

15. [直方图反向投影Back projection](#)

16. [噪声](#)

17. [去噪声](#)

18. [滤波器](#)

    - 高通滤波(HBF)
    - 低通滤波(LPF)

19. [图像积分图算法](#) （待新增）

20. [边缘检测](#)

    - 一阶导数
      - Sobel 算子
      - Rober / Prewitt 算子
    - 二阶导数
      - Laplacian 拉普拉斯 算子
      - Canny 检测

21. [图像金字塔](#)

    - 高斯图像金字塔
    - 拉普拉斯图像金字塔

22. [图像模板匹配/识别](#)

23. [轮廓检测](#)

    - [方法](#)

24. [图像二值化](#)

25. [二值图像联通组件 ConnectedComponent](#)

26. [二值图像轮廓检测 FindContour](#)

    - FindContour
    - 绘制矩形框住物件

27. [二值图像 矩阵面积与周长](#)

28. [图像几何矩(image moments)](#)

    - Moments
    - HuMoments

29. [二值图像 霍夫变幻/检测](#)

    - 霍夫直线 houghlines
    - 霍夫圆 houghcircle

30. [图像形态学 侵蚀与膨胀dilate and erode](#)

    - 侵蚀与膨胀 dilate and erode

    - 开操作 Opening

31. [数据结构](#)

32. [知识点](#)

33. [问题解决](#)

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

<h3 id=>BGR/RGB 颜色互转</h3>

imwrite()函数要求图像为BGR or 灰度， 并且每个通道有一的bit
， 输出格式必须支持

例如 bmp要求通道有8位， png允许8 or 16位元

b = image[:,:,0]#得到蓝色通道

g = image[:,:,1]#得到绿色通道

r = image[:,:,2]#得到红色通道

#### BGR / RGB 互转

example：

```python
#method 1: openCV 读取格式为HWC
bgr = rgb[...,::-1]

#method 2
rgb = bgr[:, :, ::-1]
```

#### RGB to GBR

```python
gbr = rgb[...,[2,0,1]]
```

------

<h3 id=>图像Resize</h3>

```cv2.resize(src, (x, y), fx = None, fy = None, interpolation=cv.INTER_NEAREST)``` 

缩放方式可以有两种：

- 自定义像素缩放: 在x, y 的地方设定自定义的像素

- 按比例缩放：在fx, fy的地方 可以设定0.x, 假设输入0.5, 就按照原来的比例缩小50%， 如果都设为2， 就等于放大了一倍

- interpolation （插值法表达）:

  **在一维空间中，最近点插值就相当于四舍五入取整。在二维图像中，像素点的坐标都是整数，该方法就是选取离目标点最近的点。会在一定程度上损失 空间对称性（Alignment），  插值法能够帮助图片在缩放保持比较好的图像品质**

  - INTER_NEAREST	最近邻插值
  - INTER_LINEAR	双线性插值（默认设置）
  - INTER_AREA	使用像素区域关系进行重采样。 它可能是图像抽取的首选方法，因为它会产生无云纹理的结果。 但是当图像缩放时，它类似于- - - INTER_NEAREST方法。
  - INTER_CUBIC	4x4像素邻域的双三次插值
  - INTER_LANCZOS4	8x8像素邻域的Lanczos插值

  

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

Image[y, x]可以直接返回坐标的像素值

例如创造了一条直线在histImg上 起始点是(x, y) = (0, 125这个位置)， 如下图可以清楚看见图像， 则我们要确认图像上的蓝色线的起始点像素值， 则可以histImg[125, 0]， 注意这里是[y, x], 得到BGR返回样式， 如果是灰度图 返回的就是单个值

```python
histImg = np.zeros([256,256,3], np.uint8)
cv.line(histImg, (0,125), (255, 255), [240, 0, 0])
# cv.imshow("line", histImg)
# cv.waitKey()
# cv.destroyAllWindows()

>> histImg[125, 0]
array([240,   0,   0], dtype=uint8)
```

![image-20190701192716287](https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/line.png?raw=true)





image[Row-min:max, Column-min:max, BGR[索引]]``` : 先行后列

**example:**

```
src1 = np.zeros(shape=[400, 400, 3]， dtype=np.uint8)
src1[100:200, 100:200, 0] = 255
```



<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/square.png?raw=true" width="400">



------

<h3 id=>逻辑运算</h3>

```python
cv2.bitwise_and(src1, src2) :取重叠的像素A跟B都有的值
cv2.bitwise_xor(src1, src2) : A跟B各自的加上重叠后的值
cv2.bitwise_or(src1, src2): 重叠的像素不是A就是B
cv2.bitwise_not(src):not表示都不属于集合里面的元素，所有255-原来的值 = 反向后的值
```





------

<h3 id=>ROI区域操作(mask)</h3>

Example ： 从原图获取ROI的位置, 改变ROI颜色

```python
src = cv.imread("spider-man2.jpg")
cv.imshow("input", src)
h, w = src.shape[:2]

#获取ROI
cy = h//2 
cx = w//2
roi = src[cy-100:cy+100, cx-100:cx+100]
# cv.imshow("roi", roi)

#copy ROI
image = np.copy(roi)

#modify ROI
roi[:, :, 0] = 0
cv.imshow("result", src)
```

效果如下

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/spider-man.png?raw=true" width="400">



### Mask 操作

Example：

以下例子利用mask的操作完成想要的图片 1.黑色背景彩色人 2.蓝色背景彩色人

图片依序是 原图， 黑底彩人 蓝色彩人

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/girl_green_background.png?raw=true" width="200"><img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/girl_black_background.png.png?raw=true" width="200"><img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/girl_blue_background.png?raw=true" width="200">



mask制作的思路 （这里不用什么与非来解释，太过麻烦容易忘）

1.想像mask就是一块板子，透明与黑色的部分， 黑色是为了要掩盖不想要的部分

2.透明（白）是为了要让原色穿透过去的部分

例如例子中第一个需求：

1. 将二值化的图取反得到白背景/黑人的mask
2. and操作表示会显示src2 and src2两个都有的像素， 这一步就是一张正常的src2 彩人绿背景
3. 将刚刚弄好的Mask(黑底白人）放上去，那么黑色就把绿色覆盖了， 留下彩人， 也就是我们要的黑背景彩色人

```python
# example with ROI - generate mask
src2 = cv.imread("greenkid.jpg");
src2 = cv.resize(src2, None, fx=0.75, fy=0.75)
cv.imshow("src2", src2)
hsv = cv.cvtColor(src2, cv.COLOR_BGR2HSV)


"""第一个需求"""
mask = cv.inRange(hsv, (35, 43, 46), (99, 255, 255)) #將圖像二值化， 人與背景分開，人黑 背景白
cv.imshow("mask", mask)

mask = cv.bitwise_not(mask) #取反, 白變成黑， 黑變白， 人從黑變成白色
person = cv.bitwise_and(src2, src2, mask=mask) 
#1. and操作表示会显示src2 and src2两个都有的像素， 这一步就是一张正常的src2 彩人绿背景
#2. 将刚刚弄好的Mask(黑底白人）放上去，那么黑色就把绿色覆盖了， 留下彩人
cv.imshow("person", person)



"""第二个需求"""
result = np.zeros(src2.shape, src2.dtype)
result[:,:,0] = 255 #将图片全部弄成蓝色 0 表示channel 的B
cv.imshow("result", result)
# combine background + person
mask = cv.bitwise_not(mask)#取反，人变回黑色， 背景白色


dst = cv.bitwise_or(person, result, mask=mask) 

#person : 彩色人黑底
#result 全蓝
#1. or 操作表示显示所有的像素值， 到这一步会变成背景蓝色， 人也偏蓝色（因为加上result的蓝）
#2. 将黑色人， 白背景的放上去， 就会显示黑人蓝底的图 这一步完成




dst = cv.add(dst, person)
#将彩色人放上刚刚完成的黑人蓝底，覆盖上去就完成蓝底彩色人

cv.imshow("dst", dst)

cv.waitKey(0)
cv.destroyAllWindows()
```

------

<h3 id=>图像直方图统计/均衡化</h3>

定义直方图生成的函数method

### 直方图绘制

**def custom_hist**

统计图像中的像素点个数

可以靠统计出来的值，做更多的动作， 例如图像分割

处理的思路：
为了满足”统计图像中的像素点个数“的需求

1. 利用原图像的h, w 当做遍历的依据
2. 利用np.zeros创造全是0的数组， 范围是 0 - 255， 所以创造256个0数组
3. 利用刚刚的hw遍历原图每个坐标点上的像素，假设第一个遍历的像素值是L27 那就在hist[27]+1， 全部遍历完之后，hist就得到所有像素值的个数了
4. y_pos 创造0~256
5. plt.bar(left, height, alpha=1, width=0.8, color=, edgecolor=, label=, lw=3)
   1. left：x轴的位置序列，一般采用arange函数产生一个序列； 
   2. height：y轴的数值序列，也就是柱形图的高度，一般就是我们需要展示的数据； 
   3. alpha：透明度 
   4. width：为柱形图的宽度，一般这是为0.8即可； 
   5. color或facecolor：柱形图填充的颜色； 
   6. edgecolor：图形边缘颜色 
   7. label：解释每个图像代表的含义 
   8. linewidth or linewidths or lw：边缘or线的宽度
6. plt.xticks 表示的是刻度，传入的必须是数组形式包含步长 例如x
7. plt.ylabel('Frequency') / plt.xlabel('pixels') 分别代表y, x轴名称
8. plt.show 将图标呈现出来

Example：

```python
def custom_hist(gray):
    h, w =gray.shape
#     print(h, w)
    hist = np.zeros([256], dtype=np.int32)
    for row in range(h):
        for col in range(w): #利用遍历的方式，统计每个像素值的个数，
            pv = gray[row, col]
            hist[pv] += 1
    y_pos = np.arange(0, 256, 1, dtype=np.int32)
    plt.bar(y_pos, hist, align='center', color='r', alpha=0.5)
    x = range(0, 256, 20)
    plt.xticks(x)
    plt.ylabel('Frequency')
    plt.xlabel('pixels')
    plt.title('Histogram')
    plt.show()

```



<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/histogram.png?raw=true" width="400">





**def image_hist**

1. 定义color
2. 遍历这三个color (BGR)
3. cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumlate ]])
   - 其中第一个参数 images 必须用方括号括起
   - 第二个参数 channels 是用于计算直方图的通道，这里使用灰度图计算直方图，所以就直接使用第一个通道, 如果是灰度图就是[0]， BGR = [0],[1],[2], HSV = [0], [1], [2]
   - 第三个参数 要计算的ROI区域，如果是整个图像就None
   - 第四个参数 histSize 表示这个直方图分成多少份（即多少个直方柱bin） 一般如果统计[0-255], 那么就填写256
   - 第五个参数 ranges 表示要计算的像素值范围， [0, 255] 表示直方图能表示像素值从 0 到 255 的像素, 如果取H,S 则[0, 180, 0, 255]
   - 由于直方图作为函数结果返回了，所以 hist 没有意义
   - accumulate 是一个布尔值，用来表示直方图是否叠加
4. plot 用来绘制  
5. plt.xlim跟xticks有点相似, xticks 可以手刻度数

```python
def image_hist(image):
    cv.imshow("input", image)
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()    
```

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/histogramBGR.png?raw=true" width="400">



**直方图均衡化**

```cv.equalizeHist(src)``` : 可将原图自动做均衡化， 提高对比加强深度

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/histequal.png?raw=true" width="400" align="left"><img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/result.png?raw=true" width="300">

------

<h3 id=>直方图反向投影Back projection</h3>

反向投影可以用来做图像分割，寻找ROI区间。它会输出与输入图像大小相同的图像，输出的图像上每一个**像素值**代表了输入图像上对应点属于目标对象的概率，简言之，<mark>输出图像中像素值越高的点越可能代表想要查找的目标</mark>。

**执行思路**

1.对ROI区域生成Hist， 最好输入RGB or HSV图， 颜色较容易被识别

2.在将生成好的Hist（充满信息的Hist)来投射到我们的目标图上， 也就使用到

```cv2.calcBackProject()``` 



```python
def back_projection_demo():
    sample = cv.imread("sample.png")
    target = cv.imread("target.png")
    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)
    roiHist = cv.calcHist([roi_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    #0, 1 是H S
    #最后一个参数是h 0~180, S 0~256
    print(roiHist)
    print(roiHist.shape)
		#roiHist = np.bincount(roiHist)
    print(roiHist)
    plt.plot(roiHist)
    cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX)
    #cv2.NORM_MINMAX 对数组的所有值进行转化,使它们线性映射到最小值和最大值之  间
    dst = cv.calcBackProject([target_hsv], [0, 1], roiHist, [0, 180, 0, 256], 1)
    cv.imshow("BP", dst)     

def hist2d_demo(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    dst = cv.resize(hist, (400, 400))
    cv.imshow("image", image)
    cv.imshow("hist", dst)
    plt.imshow(hist, interpolation='nearest')
    plt.title("2D Histogram")
    plt.show()    
 
x = cv.imread("target")    
back_projection_demo()
hist2d_demo(x)

cv.waitKey()
cv.destroyAllWindows()
```

------

<h3 id=>BGR/RGB 互转</h3>

------

<h3 id=>噪声 noise</h3>

- #### 椒盐噪声（salt-pepper)

  - 椒盐噪声 = 椒噪声 + 盐噪声 ，椒盐噪声的值为0(黑色)或者255(白色)，这里假设为等概率的出现0或者255。

  为图像添加椒盐噪声的的步骤如下：

  1. 依SNR制作mask，用于判断像素点是原始信号，还是噪声
  2. 依mask给原图像赋噪声值

  参考博文：https://blog.csdn.net/u011995719/article/details/83375196

  ```python
  def addsalt_pepper(img, SNR):
      img_ = img.copy()
      c, h, w = img_.shape
      mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
      mask = np.repeat(mask, c, axis=0)     # 按channel 复制到 与img具有相同的shape
      img_[mask == 1] = 255    # 盐噪声
      img_[mask == 2] = 0      # 椒噪声
  
      return img_
  img = cv2.imread('spider-man2.jpg')
  
  
  #以下是单独处理一张图像
  img = addsalt_pepper(img.transpose(2, 1, 0), 0.9)
  img = img.transpose(2, 1, 0)
  img = cv2.imwrite("spider-man-noise.jpg", img)
  ```

- #### 高斯噪声 （Gaussian）

  - 高斯噪声是指它的概率密度函数服从高斯分布（即正态分布）的一类噪声， 椒盐噪声是出现在随机位置、噪点深度基本固定的噪声，高斯噪声与其相反，是几乎每个点上都出现噪声、噪点深度随机的噪声。
  - 通过概率论里关于正态分布的有关知识可以很简单的得到其计算方法，高斯噪声的概率密度服从高斯分布（正态分布）其中有means（平均值）和sigma（标准方差）两个参数
  - <img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/gaussian_formula.png?raw=true" width="400">

  ```python
  def gaussian_noise(image):
      noise = np.zeros(image.shape, image.dtype) #取得原图像的大小制作成像素值都为0的数字图像
      m = (15, 15, 15)
      s = (30, 30, 30)
      cv.randn(noise, m, s) #randn填充高斯分布
      dst = cv.add(image, noise)
      cv.imshow("gaussian noise", dst)
      return dst
  
  src = cv.imread("spider-man2.jpg")
  src.dtype
  gaussian_noise(src)
  
  cv.waitKey(0)
  cv.destroyAllWindows()
  ```

------

<h3 id=>去噪声</h3>

以下是去噪的方式， 在滤波器篇幅会作详细介绍

- 均值去噪声
- 高斯模糊去噪声
- 中值模糊去噪声
- 非局部均值去噪声
  - ```cv2.fastNlMeansDenoisingColored(src, dst=None, h=None, hColor=None, templateWindowSize=None, searchWindowSize=None)```

------

<h3 id=>滤波器 - 高通滤波</h3>

检测图像某区域， 根据像素与周围像素的亮度来提升（boost)

<h3 id=>滤波器 - 低通滤波</h3>
<h4 id=>平均模糊</h4>

用卷积框覆盖区域所有像素的平均值来代替中心元素

cv2.blur(src, (kernel size))

<h4 id=>高斯模糊</h4>

```cv2.GaussianBlur(src, (kernel size), sigmaX)```

现在把卷积核换成高斯核，简单的说方框不变，将原来每个方框的值是相等的，现在里面的值是符合高斯分布的，方框中心的值最大，其余方框根据距离中心元素的距离递减，构成一个高斯小山包，原来的求平均数变成求加权平均数，权就是方框里的值。实现的函数是cv2.GaussianBlur()。需要指定高斯核的宽和高（必须是奇数），以及高斯函数沿X,Y方向的标准差。如果我们只指定了X方向的标准差，Y方向也会取相同值，如果两个标准差都是0.那么函数会根据核函数的大小自己计算，高斯滤波可以有效的从图像中去除高斯噪音。
也可以使用cv2.getGaussianKernel()自己构建一个高斯核。



需要注意的是kernel size大小， 数值越大越模糊, sigma值也能够影响模糊程度

<h4 id=>中值滤波</h4>

dst = cv.medianBlur(src, 3)```：大小只能设定奇数，数值越大强度越强

就是用与卷积框对应像素的中值来替代中心像素的值，这个滤波器经常用来去除椒盐噪声。前面的滤波器都是用计算得到的一个新值来取代中心像素的值，而中值滤波是用中心像素周围或者本身的值来取代他，他能有效去除噪声。卷积核的大小也应该是一个奇数。需要给原始图像加上50%的噪声，然后用中值模糊。

###  

### 边缘保留滤波

图像卷积处理无论是均值还是高斯都是属于模糊卷积，都有一个共同的特点就是模糊之后图像的边缘信息受到了破坏。边缘保留模糊滤波方法有能力通过卷积处理实现图像模糊的同时对图像边缘不造成破坏，可以更加完整的保存了图像整体边缘（轮廓）信息，我们称这类滤波算法为边缘保留滤波算法（EPF）。最常见的边缘保留滤波算法有以下几种

<h4 id=>双边模糊</h4>

```cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, dst=None, borderType=None)```

- src ： 原图
- d : 过滤期间使用的各像素邻域的直径
- sigmaColor : 色彩空间的sigma参数，该参数较大时，各像素邻域内相距较远的颜色会被混合到一起，从而造成更大范围的半相等颜色
- sigmaSpace : 坐标空间的sigma参数，该参数较大时，只要颜色相近，越远的像素会相互影响

<h4 id=>均值迁移模糊 mean-shift(待新增）</h4>

```cv2.pyrMeanShiftFiltering(src, sp, sr, dst=None, maxLevel=None, termcrit=None)```



<h3 id=>自定义滤波器 filter2D</h3>

OpenCV提供该函数可以自行设定卷积

```def filter2D(src, ddepth, kernel, dst=None, anchor=None, delta=None, borderType=None)```

Example: 我们定义一个锐化用的卷积，可以让图像细节跟明显

```python
src = cv.imread("face.jpg")
# cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
#openCV 调用filter2D API
shaperen_op = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], np.float32)

dst1 = cv.filter2D(src, -1, shaperen_op)
cv.imshow("shape=3x3", dst1)


cv.waitKey(0)
cv.destroyAllWindows()

```

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/face.png?raw=true">



<h3 id=>图像积分图算法</h3>

待新增

------

<h3 id=>边缘检测滤波函数</h3>

### 一阶导数类

#### Sobel 算子

```Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])```

- src：输入图像
- ddepth: 输出图像的深度（可以理解为数据类型），-1表示与原图像相同的深度
- dx,dy:当组合为dx=1,dy=0时求x方向的一阶导数，当组合为dx=0,dy=1时求y方向的一阶导数（如果同时为1，通常得不到想要的结果）
- ksize:（可选参数）Sobel算子的大小，必须是1,3,5或者7,默认为3。求X方向和Y方向一阶导数时，卷积核分别为：

<img src="https://wx2.sinaimg.cn/mw1024/006GmCpQgy1g4y57qs6e9j30u0140k71.jpg" width=200>



- scale:（可选参数）将梯度计算得到的数值放大的比例系数，效果通常使梯度图更亮，默认为1
- delta:（可选参数）在将目标图像存储进多维数组前，可以将每个像素值增加delta，默认为0
- borderType:（可选参数）决定图像在进行滤波操作（卷积）时边沿像素的处理方式，默认为BORDER_DEFAULT

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/Sobel_kernel.jpg?raw=true">







Example:

```python
"""
執行思路

求出src原圖像的x以及y的梯度(dx, dy)，x以及y計算出的結果會不同， 一個是水平，一個是縱向的效果，y對圖像上水平的邊緣特別明顯， x則相反

x_grad/y_grad可能會非常大，所以必須進行convertScale都變成正值

然後在將x_grad 與 y_grad進行add 求的最終結果， 也就是dx+dy

"""

src = cv.imread("face.jpg")
cv.imshow("src", src)

h, w = src.shape[:2]
x_grad = cv.Sobel(src, cv.CV_32F, 1, 0)
# x_grad = cv.Scharr(src, cv.CV_32F, 1, 0) #也可以用scharr算子試試
#第三個參數1 
y_grad = cv.Sobel(src, cv.CV_32F, 0, 1) 
# y_grad = cv.Scharr(src, cv.CV_32F, 0, 1) 

x_grad = cv.convertScaleAbs(x_grad)
y_grad = cv.convertScaleAbs(y_grad)
cv.imshow("x_grad", x_grad)
cv.imshow("y_grad", y_grad)

dst = cv.add(x_grad, y_grad, dtype = cv.CV_16S)
dst = cv.convertScaleAbs(dst)
cv.imshow("gradient", dst)


#下面將src跟dst合併
result = np.zeros([h, w*2, 3], dtype = src.dtype)
result[0:h, 0:w, :] = src
result[0:h, w:2*w, :] = dst
cv.imshow("result", result)
# cv.imwrite("")

cv.waitKey(0)
cv.destroyAllWindows()
```

Result:

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/Sobel_result.jpg?raw=true">





convertScaleAbs**

sobel or scharr 算出來的值會很大 容易超出0~255, 利用convertScale 將元素都變成正值无符号8bit的形式

```convertScaleAbs(src[, dst[, alpha[, beta]]])```

- src 輸入圖像
- dst 目標圖像
- alpha optional scale factor.
- beta optional delta added to the scaled values.





#### Robert / Prewitt 算子

利用cv.filter2D的方式对原图进行卷积操作



Example:

```python
import cv2 as cv
import numpy as np

src = cv.imread("face.jpg")

#robert 定義2*2的
robert_x = np.array([[1, 0], 
                     [0, -1]], dtype=np.float32)
robert_y = np.array([[0, -1], 
                     [1, 0]], dtype=np.float32)

#prewitt 定義為3*3的
prewitt_x = np.array([[-1, 0,  1], 
                      [-1, 0, 1],  
                      [-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[-1, -1, -1], 
                      [0, 0, 0], 
                      [1, 1, 1]],  dtype=np.float32)

#利用filter2D 來進行卷積

robert_grad_x = cv.filter2D(src, cv.CV_16S, robert_x)
robert_grad_y = cv.filter2D(src, cv.CV_16S, robert_y)
robert_grad_x = cv.convertScaleAbs(robert_grad_x)
robert_grad_y = cv.convertScaleAbs(robert_grad_y)

prewitt_grad_x = cv.filter2D(src, cv.CV_32F, prewitt_x)  #取值空間比較大所以給CV_32F
prewitt_grad_y = cv.filter2D(src, cv.CV_32F, prewitt_y)
prewitt_grad_x = cv.convertScaleAbs(prewitt_grad_x)
prewitt_grad_y = cv.convertScaleAbs(prewitt_grad_y)

# cv.imshow("robert x", robert_grad_x);
# cv.imshow("robert y", robert_grad_y);
# cv.imshow("prewitt x", prewitt_grad_x);
# cv.imshow("prewitt y", prewitt_grad_y);

h, w = src.shape[:2]
robert_result = np.zeros([h, w*2, 3], dtype=src.dtype)
robert_result[0:h,0:w,:] = robert_grad_x
robert_result[0:h,w:2*w,:] = robert_grad_y
cv.imshow("robert_result", robert_result)

prewitt_result = np.zeros([h, w*2, 3], dtype=src.dtype)
prewitt_result[0:h,0:w,:] = prewitt_grad_x
prewitt_result[0:h,w:2*w,:] = prewitt_grad_y
cv.imshow("prewitt_result", prewitt_result)


cv.waitKey(0)
cv.destroyAllWindows()
```







###  

### 二阶导数类

#### Laplacian 拉普拉斯 算子

图像的一阶导数算子可以得到图像梯度局部梯度的相应值，那么二阶导数能通过瞬间图像像素值强度的变化来检测图像边缘，其原理跟图像的一阶导数有点类似，只是在二阶导数是求X、Y方向的二阶偏导数，对图像来说：

- X方向的二阶偏导数就是 dx = f(x+1, y) + f(x-1, y) – 2*f(x, y)
- Y方向的二阶偏导数就是 dy = f(x, y+1) + f(x, y-1) – 2*f(x, y)

```Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]])```

- src：源图像 
- dst：目标图像。 
- ddepth：输出图像的深度。 
- ksize：用于计算二阶导数的滤波器的孔径尺寸，大小必须为正奇数，且有默认值1。 
- scale：计算导数值时可选的缩放因子，默认值是1。 
- delta：有默认值可忽略。 
- borderType：有默认值可忽略。

Example:

```python
import cv2 as cv
import numpy as np

image = cv.imread('face.jpg')
h, w = image.shape[:2] 
src = cv.GaussianBlur(image, (0, 0), 1)
dst = cv.Laplacian(src, cv.CV_32F, ksize=3, delta=127)
dst = cv.convertScaleAbs(dst)



result = np.zeros([h, w*2, 3], dtype=image.dtype)
result[0:h,0:w,:] = image
result[0:h,w:2*w,:] = dst
cv.imshow("result", result)
# cv.imwrite("D:/laplacian_08.png", result)

cv.waitKey()
cv.destroyAllWindows()
```

Result

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/Laplacian_result.jpg?raw=true">:









#### Canny 检测

Canny 边缘检测算法 是 John F. Canny 于 1986年开发出来的一个多级边缘检测算法，也被很多人认为是边缘检测的 最优算法, 最优边缘检测的三个主要评价标准是:

- 低错误率: 标识出尽可能多的实际边缘，同时尽可能的减少噪声产生的误报。
- 高定位性: 标识出的边缘要与图像中的实际边缘尽可能接近。
- 最小响应: 图像中的边缘只能标识一次。

> Canny算法是如何做到精准的边缘提取的，主要是靠下面五个步骤

1. 高斯模糊 – 抑制噪声
2. 梯度提取得到边缘候选
3. 角度计算与非最大信号抑制
4. 高低阈值链接、获取完整边缘
5. 输出边缘

```python
cv2.Canny(src, dst, threshold1, threshold2, apertureSize, L2gradient)
```

- src：輸入圖，單通道8位元圖。
- dst：輸出圖，尺寸、型態和輸入圖相同。
- threshold1：对于任意边缘像素低于TL(低阈值)的则丢弃
- threshold2：对于任意边缘像素高于TH(高阈值）的则保留
- apertureSize ：Sobel算子的核心大小。
- L2gradient ：梯度大小的算法，預設為false。

对于任意边缘像素值在TL与TH之间的，如果能通过边缘连接到一个像素大于

TH而且边缘所有像素大于最小阈值TL的则保留，否则丢弃

Example

```python
import cv2 as cv
import numpy as np



src = cv.imread("face.jpg")
# cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

#t1 = 100, t2 = 3*3*t1 = 300
edge = cv.Canny(src, 100, 300)
cv.imshow("mask image", edge)
cv.waitKey()
cv.destroyAllWindows()
```



Result



<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/Canny_result.jpg?raw=true">



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

<h3 id=>图像金字塔</h3>

#### 高斯图像金字塔

图像金字塔概念

图像金字塔是对一张输入图像先模糊再下采样为原来大小的1/4（宽高缩小一半）、不断重复模糊与下采样的过程就得到了不同分辨率的输出图像，叠加在一起就形成了图像金字塔、所以图像金字塔是图像的空间多分辨率存在形式。这里的模糊是指高斯模糊，所以这个方式生成的金字塔图像又称为高斯金字塔图像。高斯金字塔图像有两个基本操作

- reduce 是从原图生成高斯金字塔图像、生成一系列低分辨图像
- expand是从高斯金字塔图像反向生成高分辨率图像

规则：

1. 图像金字塔在redude过程或者expand过程中必须是逐层
2. reduce过程中每一层都是前一层的1/4 (1/2 * 1/2)

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/pyramid.png?raw=true" width=500>



Example: 

```python
import cv2 as cv

#expand
def pyramid_up(pyramid_images):
    level = len(pyramid_images)
    print("level = ", level)
    for i in range(level-1, -1, -1): #step = -1 倒序 range(2, -1, -1) > 倒序循环 2 1 0
        expand = cv.pyrUp(pyramid_images[i]) #将每一个subample的重新expand
        cv.imshow("pyramid_up_" + str(i), expand)
        
#reduce 以下的定义一个下采样的过程， 图像大小从 1 > 1/4 > 1/8 > 1/16 具体采样多少次可以依照level设定
def pyramid_down(image, level=3):
    temp = image.copy()
    pyramid_images = [] #定义空list，把subsample后的img 放进list
    for i in range(level): #利用level来定义subsample的层数
        dst = cv.pyrDown(temp) #执行pyrDown
        pyramid_images.append(dst) #将subsample后的图放进list
        temp = dst.copy() #subsample后的目标图copy后 重新丢回for的第一步再一次的subsample
        cv.imshow("haha", temp)
    return pyramid_images
        
        
src = cv.imread("face.jpg")
# cv.imshow("input", src)

# pyramid_down(src)
pyramid_up(pyramid_down(src))

cv.waitKey(0)
cv.destroyAllWindows()
```



#### 拉普拉斯图像金字塔

对输入图像实现金字塔的reduce操作就会生成不同分辨率的图像、对这些图像进行金字塔expand操作，然后使用reduce减去expand之后的结果就会得到图像拉普拉斯金字塔图像。

举例如下：
    

输入图像G(0)

金字塔reduce操作生成 G(1), G(2), G(3)

拉普拉斯金字塔：

L0 = G(0)-expand(G(1))

L1 = G(1)-expand(G(2))

L2 = G(2)–expand(G(3))

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/laplacian_pyramid.jpg?raw=true">

G(0)减去expand(G(1))得到的结果就是两次高斯模糊输出的不同，所以L0称为DOG（高斯不同）、它约等于LOG所以又称为拉普拉斯金字塔。所以要求的图像的拉普拉斯金字塔，首先要进行金字塔的reduce操作，然后在通过expand操作，最后相减得到拉普拉斯金字塔图像。

Example:

```python
import cv2 as cv
import numpy as np

def pyramid_up(image, level=3):
    temp = image.copy()
    # cv.imshow("input", image)
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        # cv.imshow("pyramid_up_" + str(i), dst)
        temp = dst.copy()
    return pyramid_images

def laplacian_demo(pyramid_images):
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if (i-1) < 0:
            h, w = src.shape[:2]
            expand = cv.pyrUp(pyramid_images[i], dstsize=(w, h))
            lpls = cv.subtract(src, expand) + 127
            cv.imshow("lpls_" + str(i), lpls)
        else:
            h, w = pyramid_images[i-1].shape[:2]
            expand = cv.pyrUp(pyramid_images[i], dstsize=(w, h))
            lpls = cv.subtract(pyramid_images[i-1], expand) + 127
            cv.imshow("lpls_"+str(i), lpls)
            
src = cv.imread("face.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
# pyramid_up(src)
laplacian_demo(pyramid_up(src))

cv.waitKey(0)
cv.destroyAllWindows()            
```

------



<h3 id=>图像模板匹配/识别</h3>

简单的来说就是将我们要检测的模板放在原图上对照找到相似的像素

模板就像是卷积核，在原始图上滑动， 模板上的像素值就相当于是权重

```matchTemplate(image, templ, method[, result[, mask]]) -> result   ```

- image  原图

- templ  模板

- method 要计算匹配程度分方式

  CV_TM_SQDIEF：平方差匹配法,最好匹配为0，值越大匹配越差

  CV_TM_SQDIEF_NORMED：归一化平方差匹配法

  CV_TM_CCORR：相关匹配法，采用乘法操作，数值越大表明匹配越好

  CV_TM_CCORR_NORMED：归一化相关匹配法

  CV_TM_CCOEFF：相关系数匹配法，最好匹配为1，最差为-1

  CV_TM_CCOEFF_NORMED：归一化相关系数匹配法

```python
import cv2 as cv
import numpy as np

def template_demo():
    src = cv.imread('llk.jpg')#原图
    tpl = cv.imread("llk_tpl.png") #模板
    cv.imshow("input", src)
    cv.imshow("tpl", tpl)
    th, tw = tpl.shape[:2] #拿到模板图像大小
    result = cv.matchTemplate(src, tpl, cv.TM_CCORR_NORMED) #用归一化相关性匹配method
    
    cv.imshow("result", result)
    t = 0.98 #阈值
    loc = np.where(result > t) #输出大于阈值的

    for pt in zip(*loc[::-1]): #满足条件的绘制出来， 也就是像素匹配的才挑出来
        cv.rectangle(src, pt, (pt[0] + tw, pt[1] + th), (255, 0, 0), 1, 8, 0)
    cv.imshow("llk-demo", src)



template_demo()
cv.waitKey(0)
cv.destroyAllWindows()
```



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

<h3 id=>图像二值化-非黑即白</h3>

二值图像就是只有黑白两种颜色的图像，0 表示黑色， 1  表示白色(255) 。

常见的二值图像分析包括轮廓分析、对象测量、轮廓匹配与识别、形态学处理与分割、各种形状检测与拟合、投影与逻辑操作、轮廓特征提取与编码等。

#### OpenCV中支持的阈值操作的API如下：

```threshold(src, thresh, maxval, type[, dst]) -> retval, dst```



- src : 原图 (multiple-channel, 8-bit or 32-bit floating point).
- Thresh: 阈值设定， 以此阈值来进行0 or 255 处理， 一般设定127
- maxval maximum ：最大值 一般设定255
- type thresholding type ：阈值分割的方式
  - THRESH_BINARY = 0 二值分割
  - THRESH_BINARY_INV = 1 反向二值
  - THRESH_TRUNC = 2 截断
  - THRESH_TOZERO = 3 取零
  - THRESH_TOZERO_INV = 4 反向取零



Example:

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



# THRESH_BINARY = 0
# THRESH_BINARY_INV = 1
# THRESH_TRUNC = 2
# THRESH_TOZERO = 3
# THRESH_TOZERO_INV = 4

src = cv.imread("face.jpg")

# cv.imshow("input", src)

T = 127


gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
images = []
plt.figure(figsize=(20, 10))
for i in range(5):
    ret, binary = cv.threshold(gray, T, 255, i)
    
    images.append(binary)
    plt.subplot(1,5, i+1) 
    plt.imshow(images[i],'gray')
    titles = ['BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])


cv.waitKey(0)
cv.destroyAllWindows()



```



Result:

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/all_binary.jpg?raw=true">





#### 自适应二值化

``` adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst```

- src ：输入灰度图

- maxValue： 一般取255

- adaptiveMethod : 自适应阈值方法，如果基于盒子模糊， 就是C means, 如果基于高斯模糊，就是

- Adaptive Method ：- It decides how thresholding value is calculated.

  1. cv.ADAPTIVE_THRESH_MEAN_C : 阈值是区域内的均值
  2. cv.ADAPTIVE_THRESH_GAUSSIAN_C : 阈值是加权平均值，权重是区域内的高斯值， 权重随着距离减小

- threshold type： 阈值操作type

- blocksize:实现卷积操作的block大小，必须是奇数（便于中心化），输入图是小图 一般取25就可以

- C ：均值计算之后 减去的常数， 10左右

    

------



<h3 align = left>二值化联通组件 ConnectedComponent </h3>

#### 联通组件标记算法

扫描一副图像每个像素点，如果都是白色像素的都分为一个组，就是代表相互连通的

`connectedComponents(image，connectivity，ltype)`

- Image : 8int原图
- connectivity：4 或是 8 的邻域算法
- ltype：目标图像的深度 一般用CV_32S or CV_16U



Return : 一组output包含(num_label, 完整图像的array)



Example：

执行思路：

1. 高斯模糊
2. 二值化
3. 利用connectedComponent API 进行label标记
4. 将标记出的进行上色做出区隔

```python
def connected_components_demo(src):
    src = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow('binary', binary)
    
    output = cv.connectedComponents(binary, connectivity=8, ltype=cv.CV_32S)
    num_labels = output[0] #26个标记
    labels = output[1] #labels = 完整一副图像数组
    colors = [] #创造一个空list
    for i in range(num_labels): #随机创造出bgr颜色
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        colors.append((b, g, r))#将创造出的bgr丢进color list中， 最终遍历26次就会有26个色

    colors[0] = (0, 0, 0)
    h, w = gray.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h): #遍历每一个像素点，将刚刚创造出的color 逐像素点上色
        for col in range(w):
            image[row, col] = colors[labels[row, col]] #将刚刚创造出的color 逐像素点上色

    cv.imshow("colored labels", image)
    print("total rice : ", num_labels - 1)
    
src = cv.imread('rice.png')
# cv.imshow('src')
connected_components_demo(src)
cv.waitKey()
cv.destroyAllWindows()

```

该图返回的是带有各种颜色的米粒图



#### 联通组件状态分析

`connectedComponentsWithStats(image, connectivity, ltype)`

- Image : 8int原图
- connectivity：4 或是 8 的邻域算法
- ltype：目标图像的深度 一般用CV_32S or CV_16U

Return : 返回4个值

1. num_labels:标记的数量
2. labels:整幅图像的array
3. stats:所有被标记的矩形要素（x, y, w, h) 左上、右下、宽高、label的像素面积
4. centers:每个被标记的中心点像素

Example:

```python
def connected_components_stats_demo(src):
    src = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#     cv.imshow("binary", binary)

    num_labels, labels, stats, centers = cv.connectedComponentsWithStats(binary, connectivity=8, ltype=cv.CV_32S)
    colors = []
    for i in range(num_labels):
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        colors.append((b, g, r))

    colors[0] = (0, 0, 0)
    image = np.copy(src)
    for t in range(1, num_labels, 1):
        x, y, w, h, area = stats[t]
        cx, cy = centers[t] #取得中心位置的坐标点
        cv.circle(image, (np.int32(cx), np.int32(cy)), 2, (0, 255, 0), 2, 8, 0) #将中心位置透过circle画出
        cv.rectangle(image, (x, y), (x+w, y+h), colors[t], 1, 8, 0) #将state提供的xywh画出来
        cv.putText(image, "num:" + str(t), (x, y), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1);
        print("label index %d, area of the label : %d"%(t, area))

#     cv.imshow("colored labels", image)
    print("total rice : ", num_labels - 1)
    return binary, image


input = cv.imread("rice.png")
binary, image = connected_components_stats_demo(input)

plt.figure(figsize=(20, 10))
# plt.imshow(input)



binary = cv.cvtColor(binary, cv.COLOR_BGR2RGB)
plt.imshow(binary)
plt.figure(figsize=(20, 10))
plt.imshow(image)


cv.waitKey(0)
cv.destroyAllWindows()
```



<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/rice_connetComp.jpg?raw=true">

------

<h3 align = left> 二值图像轮廓检测FindContour </h3>
<h4 id= align = left> 轮廓检测FindContour </h4>

`findContours(image, mode, method)`

1. image：輸入圖，使用8bit單通道圖，非零的像素都會列入考虑，通常為二值後的圖。

2. mode：取得輪廓的模式。

   - CV_RETR_EXTERNAL：只取最外層的輪廓。
   - CV_RETR_LIST：取得所有輪廓，不建立階層(hierarchy)。
   - CV_RETR_CCOMP：取得所有輪廓，儲存成兩層的階層，首階層為物件外圍，第二階層為內部空心部分的輪廓，如果更內部有其餘物件，包含於首階層。
   - CV_RETR_TREE：取得所有輪廓，以全階層的方式儲存

3. method：儲存輪廓點的方法。

   - CV_CHAIN_APPROX_NONE：儲存所有轮廓点。
   - CV_CHAIN_APPROX_SIMPLE：对水平、垂直、對角線留下头尾点，假如輪廓為一矩形，只儲存對角的四個頂點

   

   返回：contours, hierarchy

   contours 会是一个列表， 列表中存储的是array的形式

   

`cv.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]`

1. image：输入输出图，输出会将图绘制在此图上
2. contours：也就是findContours()所找到的contours。
3. contourIdx：指定绘制某個輪廓
4. color：绘制的颜色
5. lineType：线条type
6. hierarchy：轮廓阶层，也就是findContours()所找到的hierarchy
7. maxLevel：最大阶层的輪廓，可以指定想要画的輪廓，有輸入hierarchy時才會考慮，輸入的值代表绘制的层数

Example:

执行思路：

1. 原图进行二值化
2. findContours找出轮廓
3. 利用找到的contour利用drawContours绘制出轮廓

```python
def threshold_demo(image):
    #去噪+二值
    dst = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
    return binary

def canny_demo(image):
    t = 100
    canny_output = cv.Canny(image, t, t*2)
    return canny_output

src = cv.imread('face.jpg')
# cv.imshow('input', src)
binary = threshold_demo(src)

contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for c in range(len(contours)):
    cv.drawContours(src, contours, c, (0, 0, 255), 2, 8)
    
original = cv.imread('face.jpg')
original = original[...,::-1]
src = src[...,::-1]
# original = cv.cvtColor(original, cv.COLOR_BGR2RGB)
# src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(original)
plt.subplot(1, 2, 2)
plt.imshow(src)
cv.waitKey(0)
cv.destroyAllWindows()
```

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/findContour.jpg?raw=true">

<h4 id= align = left> 绘制矩形框住物件 </h4>

當得到物件輪廓後，可以用以下三种方式将物件框住

- boundingRect() : 框住此轮廓的最小正矩形
- minAreaRect() : 框住此轮廓的最小斜矩形
  - 返回值:( center (x,y), (width, height), angle of rotation )
- minEnclosingCircle(): 框住此轮廓的最小圓形，

這些函数让我們填補空隙，或者作進一步的物件辨識

`cv.morphologyEx(src, op, kernel)`
该函数执行形态学转换， 通常使用在二值图
It needs two inputs, one is our original image, second one is called structuring element or kernel

最基本的两个形态转换就是腐蚀Erosion 跟膨胀Dilation

下面的例子就是利用3*3的kernel执行膨胀操作



Example:

执行思路：

1. 将原图Canny 找出边缘并且morphology膨胀
2. 在用findContours找出轮廓
3. 在看利用何种方式将轮廓给框住



```python
def canny_demo(image):
    t = 200
    canny_output = cv.Canny(image, t, t * 2)
#     cv.imshow("canny_output", canny_output)
#     cv.imwrite("D:/canny_output.png", canny_output)
    return canny_output


src = cv.imread("stuff.jpg")
# cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
# cv.imshow("input", src)
binary = canny_demo(src)
k = np.ones((3, 3), dtype=np.uint8)#弄一个3 by 3 kernel
binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)
cv.imshow('binary', binary)

# 轮廓发现
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in range(len(contours)):
#     x, y, w, h = cv.boundingRect(contours[c]);
#     cv.drawContours(src, contours, c, (0, 0, 255), 2, 8)
#     cv.rectangle(src, (x, y), (x+w, y+h), (0, 0, 255), 1, 8, 0);
    rect = cv.minAreaRect(contours[c]) #将取得的contours 用遍历的方式把每一个center x, y 都取出
    cx, cy = rect[0]
    box = cv.boxPoints(rect)
    box = np.int64(box)
    cv.drawContours(src,[box],0,(0,0,255),2)
    cv.circle(src, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 2, 8, 0)


# 显示
# cv.imshow("contours_analysis", src)
# cv.imwrite("D:/contours_analysis.png", src)

src = src[...,::-1]
plt.figure(figsize=(20, 10))
plt.imshow(src)

cv.waitKey(0)
cv.destroyAllWindows()



```



<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/minAreaRect.jpg?raw=true">

------

<h3 align = left> 二值图像矩阵面积与周长 </h3>

利用findContour找出物体的轮廓之后， 能利用`contourArea`, `arcLength`，來找轮廓的質心、周長、面积，

#### 计算面积

`cv.contourArea(contour, oriented)`:

- contour：输入的轮廓
- oriented：轮廓方向，如果設為ture的話除了面积還會記錄方向，顺时针和逆时针會有正負号的差異，預設為false，不論轮廓方向都返回正的面积值



### 计算周长（弧长）

`cv.arcLength(curve, closed)`:

- curve：输入轮廓，一個含有2維點的vector。
- closed：轮廓封閉，指定curve是否封閉，true or False

Example：

执行思路：

1. Canny检测二值化
2. 利用morphologyEx形态学膨胀将二值图处理一下
3. findContour 找出目标轮廓
4. 求出面积与周长当做阈值筛选目标
5. 绘制目标物矩形与中心点

```python
import cv2 as cv
import numpy as np

def canny_demo(image):
    t = 80
    canny_output = cv.Canny(image, t, t*2)
    cv.imshow('canny_output', canny_output)
    return canny_output

src = cv.imread('zhifangqiu.jpg')
cv.imshow('src', src)
binary = canny_demo(src)
k = np.ones((3, 3), dtype=np.uint8)
binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)



'''轮廓发现'''
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for c in range(len(contours)):
    area = cv.contourArea(contours[c])
    arclen = cv.arcLength(contours[c], True) #求出弧长
    if area < 100 or arclen < 100:#表示如果小于100, 就继续 不绘制， 过滤最小面积
        continue
#     if area > 100 or arclen > 100 :


    #由上面的周长以及面积来当做阈值， 如果小于100的 就不会是我们要找的对象
    rect = cv.minAreaRect(contours[c]) #将目标物的矩形框找出来
    cx, cy = rect[0] #去除矩形正中心点
    box = cv.boxPoints(rect)
    box = np.int0(box) #记得转换
    cv.drawContours(src, [box], 0, (0, 0, 255), 2) #将矩形绘制出来
    cv.circle(src, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 2, 8, 0)#将矩形中心点绘制出来


cv.imshow('contours_analysis', src)
cv.waitKey()
cv.destroyAllWindows()

```

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/detect_fatball.jpg?raw=true">

------

<h3 align = left> 图像几何矩 ImageMoments </h3>

### moments

对图像二值图像的每个轮廓，可以计算轮廓几何矩，根据几何矩可以计算图像的中心位置，估计得到中心位置可以计算中心矩、然后再根据中心矩可以计算Hu矩。

```cv.moment(InputArray array, bool binaryImage=false)```

返回的是 mu
这个mu 包含了m10, m01, m00
m10 表示x
m01 表示y
m00 表示area



For image with pixel intensities I(x,y), the raw image moments 𝑀𝑖𝑗 are calculated by

𝑀𝑖𝑗=∑𝑥∑𝑦𝐼(𝑥,𝑦)



关于moments可以用物理学的角度来看，假设图像上的每个像素有重量， 那么这个重量等同于本身的强度，假设I(x,y)是图像中像素(x,y)的强度.那么m(i,j)是所有可能的x和y的和：I(x,y)*(x ^ i)*(y ^ j).

Moments 总和了image的shape用：
I(x, y)

𝑀𝑖𝑗=∑𝑥∑𝑦𝐼(𝑥,𝑦)
i+j 是n阶距

因为可以用物理学来看，那么centroid表示的也就是这个物件的重心 （mass center）
那么如果要找出重心坐标 公式会是：



x、y坐标总和/所有 就是中心位置

$C_x = \frac{M10}{M00}$

$C_y = \frac{M01}{M00}$

Example：

执行思路：

1. 经过canny二值化
2. morphologyEX 膨胀处理
3. findContour 找到轮廓
4. 遍历轮廓，minAreaRect 取得最小矩形
5. 取得矩形中心x, y, width, height
6. 找出最小与最大的宽高比
7. 利用moments 找到mm
8. 利用mm求出centroid
9. cv.boxPoint求出box, 并转成 int8
10. 绘制

```python
import cv2 as cv
import numpy as np

def canny_demo(image):
    t = 80
    canny_output = cv.Canny(image, t, t * 2)
    cv.imshow('canny_output', canny_output)
    return canny_output

src = cv.imread('stuff2.jpg')
cv.imshow('input', src)
binary = canny_demo(src)
k = np.ones((3, 3), dtype=np.uint8)
binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)


contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in range(len(contours)):
    rect = cv.minAreaRect(contours[c])
    cx, cy = rect[0]
    ww, hh = rect[1]
    ratio = np.minimum(ww, hh)/np.maximum(ww, hh)
    print(ratio) #求出最小与最大的比例
    
    mm = cv.moments(contours[c])
    m00 = mm['m00']
    m10 = mm['m10']
    m01 = mm['m01']
    cx = np.int(m10/m00)
    cy = np.int(m01/m00)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    if ratio > 0.9 :
        cv.drawContours(src, [box], 0, (0, 0, 255), 2)
        cv.circle(src, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 2, 8, 0)
    if ratio < 0.4:
        cv.drawContours(src, [box], 0, (255, 0, 255), 2)
        cv.circle(src, (np.int32(cx), np.int32(cy)), 2, (0, 0, 255), 2, 8, 0)
        
cv.imshow('contours_analysis', src)
cv.waitKey()
cv.destroyAllWindows()
```



<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/moments.jpg?raw=true">



### HuMoments

对图像二值图像的每个轮廓，可以计算轮廓几何矩，根据几何矩可以计算图像的中心位置，估计得到中心位置可以计算中心矩、然后再根据中心矩可以计算胡矩·

array是输入的图像轮廓点集合
输出的图像几何矩，根据几何矩输出结果可以计算胡矩，胡矩计算的API如下：
`cv.HuMoments(mm)`

`matchShapes(contour1, contour2, method, parameter)`
可以从contour1， contour2来比较两个形状，返回一个matirc展示相似度，result的值越低表示表示相似度越高，这是基于Hu-moment的值来计算

Example:

执行思路:

- 找到原图A(被匹配的）以及一张要负责匹配的图B
- findContours 找到两张图的轮廓（输入之前必须转灰度）
- 图B 透过Moments 找到mm（几何矩）, 丢进HuMoments计算出Hu矩
- 遍历的方式将原图A的找到的轮廓都算出HuMoments(按照上一步的方式）
- 调用API matchShape来进行contour1, contour2来匹配hum1, hum2的比对，值越小越相似
- 可以设定阈值，例如dist < 0.5 就drawContours

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def contours_info(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


src = cv.imread("carnumber.jpg")
# result = src.copy()
# cv.namedWindow("input1", cv.WINDOW_AUTOSIZE)
# cv.imshow("input1", src)
# plt.imread(src)
msrc = src[...,::-1]
plt.figure(1)
plt.figure(figsize=[10, 10])
plt.subplot(1, 2, 1)
plt.imshow(msrc)
src2 = cv.imread("carnumber5.jpg")
# result2 = src2.copy()
# cv.imshow("input2", src2)
msrc2 = src2[...,::-1]

plt.subplot(1, 2, 2)
plt.imshow(msrc2)





# 轮廓发现
contours1 = contours_info(src)
contours2 = contours_info(src2) 

# 几何矩计算与hu矩计算
mm2 = cv.moments(contours2[0])#计算与匹配的moments
hum2 = cv.HuMoments(mm2) #计算出Hu矩

# 轮廓匹配
for c in range(len(contours1)):
    mm = cv.moments(contours1[c]) #求出每一个目标轮廓的mm
    hum = cv.HuMoments(mm)#求出hu矩
    dist = cv.matchShapes(hum, hum2, cv.CONTOURS_MATCH_I1, 0)#用Hu-moments计算相似度， 值越小代表越高
    if dist < 0.5: #设定阈值，
        cv.drawContours(src, contours1, c, (0, 0, 255), 2, 8)
        
        

src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
plt.figure(figsize=[10, 10])
plt.figure(3)
plt.imshow(src)

cv.waitKey(0)
cv.destroyAllWindows()
```

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/Hu_moments.jpg?raw=true">

------



------

<h3 id=> 二值图像分析 - 霍夫变化/检测</h3>

### HoughLines

霍夫的原理解释篇幅很长， 这边只介绍API参数及使用实例

参考知乎

https://www.zhihu.com/question/35268803/answer/82100453



`cv.HoughLines(
	image,
	lines,
	rho,
	theta,
	threshold,
	srn = 0,
	stn = 0,
	min_theta = 0,
	max_theta = CV_PI
)`

- Image 输入图像
- Lines 输出直线
- Rho 极坐标r得步长
- Theta角度步长
- Threshold累加器阈值
- Srn、stn多尺度霍夫变换时候需要得参数，经典霍夫变换不需要
- min_theta 最小角度
- max_theta最大角度

返回 共线的rho， theta



Houghline的执行过程

1. 使用边缘检测算法（如canny算子）得到边缘检测的二值图像
2. 扫描整个图像的前景点（边缘点），遍历整个Theta的范围得出对应的Rho值，并在对应的累加器单元加1，主要进行范围的调整
3. 寻找累加器中最大值(出现频率最多的)
4. 结合所给阈值算出判断累加器共线的最小值阈值
5. 非极大值抑制
6. 得到共线的（Theta，Rho），还原到直角坐标系，并在图像显示 



Example：

执行思路

1. 利用canny检测取得边缘二值图像
2. 调用HoughLines取得线段Lines 包含
3. 遍历的方式取出每一个Line的tho, theta
4. a = np.cos(theta), b = np.sin(theta)
5. X0 = a*rho
6. Y0 = b*rho
7. 求出pt1, pt2坐标
8. 绘制直线



```python
import cv2 as cv
import numpy as np

def canny_demo(image):
    t = 80
    canny_output = cv.Canny(image, t, t*2)
    return canny_output

src = cv.imread('numbersarray.jpg')

binary = canny_demo(src)
lines = cv.HoughLines(binary, 1, np.pi / 180, 150, None, 0, 0)
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a* rho
        y0 = b*rho
        pt1 = (int(x0+1000 * (-b)), int(y0+1000*(a))) #pt1 表示直线起始坐标1000是为了将线段加长 
        pt2 = (int(x0-1000 * (-b)), int(y0 - 1000*(a))) # 终点坐标
        
        cv.line(src, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
        
cv.imshow('houghline', src)
cv.waitKey()
cv.destroyAllWindows()


```



<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/houghline.png?raw=true">



### HoughLinesP

`cv.HoughLinesP(img, rho, theta, minLineLength, maxLineGap)`

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

二值图像分析 – 霍夫圆检测
根据极坐标,圆上任意一点的坐标可以表示为如上形式, 所以对于任意一个圆, 假设中心像素点p(x0, y0)像素点已知, 圆半径已知,则旋转360由极坐标方程可以得到每个点上得坐标同样,如果只是知道图像上像素点, 圆半径,旋转360°则中心点处的坐标值必定最强.这正是霍夫变换检测圆的数学原理

`cv2.HoughCircles(image, 
method, 
dp, 
minDist, 如果设为0， 就会检测出很多通讯员， 如果是10表示两个同心圆距离是10才可以，小于10认为是同个圆，
circles=None, 
param1=None, 
param2=None, 累加到一定程度才能算圆，所以越小检测到圆越多
minRadius=None, 
maxRadius=None)`

- image: 单通道图像8位。如果使用彩色图像，需要先转换为灰度图像。
- method：定义检测图像中圆的方法。目前唯一实现的方法是cv2.HOUGH_GRADIENT。
- dp：累加器分辨率与图像分辨率的反比。dp获取越大，累加器数组越小。 一般取2以上  
- minDist：检测到的圆的中心，（x,y）坐标之间的最小距离。如果minDist太小，则可能导致检测到多个相邻的圆。如果minDist太大，则可能导致很多圆检测不到。最小必须是10，
- param1：用于处理边缘检测的梯度值方法。 边缘提取的高阈值
- param2：cv2.HOUGH_GRADIENT方法的累加器阈值。阈值越小，检测到的圈子越多。
- minRadius：半径的最小大小（以像素为单位）。
- maxRadius：半径的最大大小（以像素为单位）。

Example:

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
src = cv.imread("coin.jpg")
plt.figure(figsize=[15, 15])
plt.subplot(131)
plt.imshow(src)
canny = cv.Canny(src, 200, 300)
plt.subplot(132)
plt.imshow(canny)
dp = 2 #与原图大小相比，2比较常用
param1 = 100
param2 = 90


show_circle = src
circles = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, dp, 80, None, param1, param2, 30, 100)
if circles is not None:
    for c in circles[0,:]:
        print(c)
        cx, cy, r = c
        cv.circle(show_circle, (cx, cy), 2, (0, 255, 0), 10, 8, 0)
        cv.circle(show_circle, (cx, cy), r, (0, 0, 255), 5, 8, 0)


plt.subplot(133)
plt.imshow(show_circle[:, :, ::-1])
cv.waitKey(0)
cv.destroyAllWindows()


```



## <img src = "https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/houghcircle.jpg?raw=true">

<h3 id=> 图像形态学 侵蚀与膨胀dilate and erode </h3>

#### dilate and erode

形态学主要用于二值化后的影像，根据使用者的目的，用来凸显影像的形状特征，像边界和连通区域等，同时像细化，像素化，修剪毛刺等技术也常用于图像的预处理和后处理，形态学操作的结果除了影像本身，也和结构元素的形状有关，结构元素和空间域操作的滤波概念类似，如以下即为一个3×3的结构元素，我们可以自行决定大小和形状，在实际的使用上，是以奇数的矩形如3×3,5×5,7×7较常见。

1.膨胀可以看成是最大值滤波，即用最大值替换中心像素点
2.腐蚀可以看出是最小值滤波，即用最小值替换中心像素点。

#### 腐蝕

腐蝕顾名思义就是消融物体的边界，如果物体大于结构元素，侵蚀的结果是让物体瘦一圈，而这一圈的宽度是由结构元素大小决定的，如果物体小于结构元素，则侵蚀后物体会消失，如果物体之间有小于结构元素的细小连通，侵蚀后会分裂成两个物体

`cv2.erode(src, kernel, anchor, iterations `

`cv2.dilate(src, kernel, anchor, iterations`

- src：輸入圖，可以多通道，深度可為CV_8U、CV_16U、CV_16S、CV_32F或CV_64F。
- kernel：結構元素，如果kernel=Mat()則為預設的3×3矩形，越大侵蝕效果or膨脹越明顯。
- dst : 目標圖
- anchor：原點位置，預設為結構元素的中央。
- iterations：執行次數，預設為1次，執行越多次侵蝕效果越明顯



如下图蓝色的框的值就会被替换

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/dilate.erode.formula.jpg?raw=true">

Example

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

src = cv.imread("wangzuxian.jpg")

se = np.ones((3, 3), dtype=np.uint8)
dilate = cv.dilate(src, se, None,  (-1, -1), 1)
erode = cv.erode(src, se,  None, (-1, -1), 1)

plt.figure(figsize=[20, 10])
plt.tight_layout
plt.subplot(2, 3, 1, title="original")
plt.imshow(src[:, :, ::-1])
plt.subplot(2, 3, 2, title="dilate")
plt.imshow(dilate[:, :, ::-1])
plt.subplot(2, 3, 3, title="erode")
plt.imshow(erode[:, :, ::-1])

se_strong = np.ones((7, 7), dtype=np.uint8)
dilate_strong = cv.dilate(src, se_strong, None,  (-1, -1), 1)
plt.subplot(2, 3, 5, title="dilate_strong")
plt.imshow(dilate_strong[:, :, ::-1])

erode_strong = cv.erode(src, se_strong, None,  (-1, -1), 1)
plt.subplot(2, 3, 6, title="erode_strong")
plt.imshow(erode_strong[:, :, ::-1])

```



<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/dilate.erode.jpg?raw=true">



#### 开操作 Opening

开运算可以使物体轮廓变得光滑，还能使狭窄的连结断开，以及消除外观上的毛刺，但在物体大于结构元素的情况下，开运算与侵蚀并不相同，图像的轮廓并没有产生整体的收缩，物体位置也没有发生任何变化，假如我们对一幅影像重复进行开运算，不会产生任何变化，这点和重复进行侵蚀会加强程度的现象不同

`cv.getStructuringElement(shape, kernel, anchor = (-1, -1)` 

- shape：模板形狀

  有MORPH_RECT、MORPH_ELLIPSE、MORPH_CROSS三種可選。

- ksize：模板尺寸

`cv.morphologyEx(src, op, kernel)`

- src：输入图，可以多通道，深度可为CV_8U，CV_16U，CV_16S，CV_32F或CV_64F。
- op：操作种类，决定要进行何种型态学操作，在闭运算时输入MORPH_CLOSE。
- kernal size：结构元素。
- anchor：原点位置，预设为结构元素的中央。
- iteration：执行次数，预设为1次。

执行思路：

1.转换为灰度图

2.二值化

3.建构结构模板(getStructuringElement)

4.进行开操作

Example：

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


src = cv.imread("boost.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
dst = cv.GaussianBlur(gray, (9, 9), 2, 2)
binary = cv.adaptiveThreshold(dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 45, 15)
# binary = cv.Canny(src, 100, 200)

se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5), (-1, -1))
binary_opening = cv.morphologyEx(binary, cv.MORPH_OPEN, se)

plt.figure(figsize=[20, 10])
plt.subplot(131, title='orginal')
plt.imshow(src)
plt.subplot(132, title='binary')
plt.imshow(binary)
plt.subplot(133, title='binary_opening')
plt.imshow(binary_opening)
```

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/morphology_opening.jpg?raw=true">

------









------

<h3 id=> 数据结构 </h3>

```
Type:类型 CV_[位数][带符号与否][类型前缀][通道数]
例:CV_8UC3表示使用8位的unsighed char类型.每个像素由三个元素组成的三通道.
类型汇总:
CV_8U  (8 bit 无符号整形0~255) CV_8UC1 (1通道)  CV_8UC2 (2通道)  CV_8UC3 (3通道)  CV_8UC4(4通道) 
CV_8S   (8 bit有符号整形-128~127)
CV_8SC1 (1通道)  CV_8SC2 (2通道)   CV_8SC3 (3通道)  CV_8SC4 (4通道)   

CV_16U  (16 bit 无符号整形0~65535)
CV_16UC1 (1通道)  CV_16UC2 (2通道)   CV_16UC3 (3通道)   CV_16UC4 (4通道)   

CV_16S  (16 bit 有符号整形-32768~32767)
CV_16SC1(1通道)   CV_16SC2(2通道)   CV_16SC3(3通道)   CV_16SC4(4通道)   

CV_32S  (32 bit 有符号整形-2147483648~2147483647)
CV_32SC1   CV_32SC2    CV_32SC3  CV_32SC4   

CV_32F  (32 bit 浮点)
CV_32FC1   CV_32FC2  CV_32FC3   CV_32FC4  

CV_64F   (64 bit 浮点)
CV_64FC1   CV_64FC2  CV_64FC3  CV_64FC4  
```

------

<h3 id=> 知识点 </h3>

#### blob

Blob在机器视觉中是指图像中的具有相似颜色、纹理等特征所组成的一块连通区域。**Blob**分析就是对前景/背景分离后的二值图像，对这一块连通区域进行几何分析得到一些重要的几何特征，例如：区域的面积、中心点坐标、质心坐标、最小外接矩形、主轴等

Blob分析的一般步骤：

（1）图像分割：分离出前景和背景

（2）连通性分析：根据目标的连通性对目标区域进行标记，或者叫拓扑性分析

（3）特征量计算：描述了区域的几何特征，这些几何特征不依赖与灰度值

------

<h3 align = center> 问题解决 </h3>

1. 问题一

FindContours support only 8uC1 and 32sC1 images in function cvStartFindContours

只支持 single channel, unit8

1. 解决
   输入findContours函数之前， 先将图像进行转换成unit8, channel = 1的

```cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)```
