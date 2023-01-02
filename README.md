# 基于地理位置的社区挖掘与预测

数据：用户的兴趣地点

任务：

1. 时空轨迹可视化
2. 社团划分可视化
3. 预测

数据集：

1. Foursquare https://www.kaggle.com/chetanism/foursquare-nyc-and-tokyo-checkin-dataset 
   1. 用户：`userId`
   2. 地点分类：`venueCategoryId`, `venueCategory`
   3. 地点：`venueId`, `latitude`, `longitude`
   4. 时间：`timezoneOffset`，`utcTimestamp`
2. Brightkite http://snap.stanford.edu/data/loc-Brightkite.html 
   1. 用户：`user`
   2. 用户间的交互
   3. 地点：`location id`, `latitude`, `longitude`
   4. 时间：`check-in time`

信息：

- 用户和地点交互信息
- 地理信息
- 时序信息：工作日，小时

思路：

- 二分图：以经纬度为参照 使用前馈神经网络 将高维编码投影到二维
- 两张图
- Deepwalk: [github](https://github.com/phanein/deepwalk), [paperswithcode](https://paperswithcode.com/paper/deepwalk-online-learning-of-social#code)
- RNN/Transformer

参考文献：https://www.cse.cuhk.edu.hk/irwin.king/_media/presentations/p325.pdf 

参考网页：

1. [图深度学习](https://blog.csdn.net/qq_34539676/article/details/125608802)
2. [DeepWalk](https://zhuanlan.zhihu.com/p/56380812), [LINE](https://zhuanlan.zhihu.com/p/56478167) 含代码
3. [POI可视化](https://zhuanlan.zhihu.com/p/165095864)

Git使用：

- 拉取代码：`git pull origin main`
- 推送代码：
  - 先加到缓存：`git add emb2coord.py`
  - 可查看状态：`git status -s`
  - 再加到仓库：`git commit -m 'Add emb2coord'`
  - 或一步到位：`git commit -am 'Add emb2coord'`
  - 最后：`git push origin main`