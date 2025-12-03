import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # 若用FCM，可使用`fcmeans`库（需先安装：pip install fuzzy-c-means）
from fcmeans import FCM
import cv2
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')
# 1. 读取图像
img = cv2.imread("./data/cherry.jpg")  # 替换为你的图像路径
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB通道

# 2. 重塑图像为像素矩阵
pixels = img_rgb.reshape(-1, 3)  # 形状为 (像素总数, 3)
print(pixels.shape)

# 3. 模糊C均值聚类（FCM）
n_clusters = 2  # 樱桃和叶子两类
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(pixels)
labels = gmm.predict(pixels)  # 每个像素的聚类标签

# 4. 重建分割图像
segmented = labels.reshape(img_rgb.shape[:2])

# 5. 可视化结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(segmented, cmap="viridis")  # 用颜色区分两类
plt.title("FCM Segmentation (Cherry vs Leaf)")
plt.axis("off")

plt.show()