# OpenCTR
OpenCTR 使用 TensorFlow 和 PyTorch 实现了业内几乎所有点击率预测模型，模型测试采用 Criteo 数据集，代码注释采用中文，更适合中国用户。

## CTR 简介

点击率（CTR, Click-Through Rate）是衡量在线广告、链接或任何数字内容吸引力的重要指标。它表示用户在看到某个链接或广告后实际点击的比例，用百分比表示。CTR 是数字营销、广告投放、电子商务和搜索引擎优化中常用的关键性能指标 (KPI)。

计算公式：

$$
CTR = \frac{\text{(Clicks)}}{\text{(Impressions)}} \times 100\%
$$

- 点击次数 (Clicks)：用户实际点击广告或链接的次数。
- 展示次数 (Impressions)：广告或链接被用户看到的总次数。

## 模型简介

| 模型 | 论文 |
| --- | --- |
| LR(Linear Regression) |[统计学习方法 1805](https://ai.renyuzhuo.cn/books/StatisticsLi/CH-StatisticsLi.pdf) |
| FM(Factorization Machines) |[因子分解机 2010](https://analyticsconsultores.com.mx/wp-content/uploads/2019/03/Factorization-Machines-Steffen-Rendle-Osaka-University-2010.pdf) |
| WD(Wide & Deep) |[DLRS 2016] [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792) |
| DeepFM| [IJCAI 2017] [A Factorization-Machine based Neural Network for CTR Prediction](https://www.ijcai.org/proceedings/2017/0239.pdf)

