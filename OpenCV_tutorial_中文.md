<h1 align = center >OpenCV3 python实践</h1>
<h4 align = right >update 2019.7.7</h4>

1. [requirement](#)
2. [图像阵列算数运算](#)
3. [图像Resize与插值法](#)
4. [图像亮度与对比](#)(待新增)
5. [图像色彩空间转换](#)
6. [图像像素统计](#)
7. [图像的均值及标准差](#) (待新增)
8. [图像归一化](#)
9. [ LUT查找表applyColormap](#)
10. [图像像素点索引](#)
11. [逻辑运算](#)
12. [ROI区域操作](#)
13. [图像直方图统计/均衡化](#)
14. [直方图反向投影Back projection](#)
15. [知识点](#)
16. [噪声](#)
17. [去噪声](#)
18. [滤波器](#)
    - 高通滤波(HBF)
    - 低通滤波(LPF)
19. [图像积分图算法](#) （待新增）
20. [边缘检测](#)
21. [轮廓检测](#)
    - [方法](#)
22. [直线检测](#)
    - Hough 霍夫
23. [图像二值化](#)
24. [数据结构](#)
25. [问题解决](#)

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



<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/square.png?raw=true" style="zoom:50%"/>



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

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/spider-man.png?raw=true" style="width:400px" align="left">



### Mask 操作

Example：

以下例子利用mask的操作完成想要的图片 1.黑色背景彩色人 2.蓝色背景彩色人

图片依序是 原图， 黑底彩人 蓝色彩人

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/girl_green_background.png?raw=true" style="width:200px" align="left"><img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/girl_black_background.png.png?raw=true" style="width:200px" align="center"><img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/girl_blue_background.png?raw=true" style="width:200px">



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



<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/histogram.png?raw=true" style="width:400px">





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

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/histogramBGR.png?raw=true" style="width:400px">



**直方图均衡化**

```cv.equalizeHist(src)``` : 可将原图自动做均衡化， 提高对比加强深度

<img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/histequal.png?raw=true" style="width:400px" align="left"><img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/result.png?raw=true" style="width:300px">

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

<h3 id=>知识点</h3>

imwrite()函数要求图像为BGR or 灰度， 并且每个通道有一的bit
， 输出格式必须支持

例如 bmp要求通道有8位， png允许8 or 16位元

b = image[:,:,0]#得到蓝色通道

g = image[:,:,1]#得到绿色通道

r = image[:,:,2]#得到红色通道

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
  - <img src="https://github.com/Stephenfang51/OpenCV_tutorial_chinese/blob/Stephenfang51-patch-1/images/gaussian_formula.png?raw=true" style="width:300px">

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

<h3 align = left> 数据结构 </h3>

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

<h3 align = center> 问题解决 </h3>

1. 问题一

FindContours support only 8uC1 and 32sC1 images in function cvStartFindContours

只支持 single channel, unit8

1. 解决
   输入findContours函数之前， 先将图像进行转换成unit8, channel = 1的

```cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)```
