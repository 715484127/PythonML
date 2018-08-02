import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)  # 显示所有数据
filename = 'listings.csv'
features = [
    'accommodates',
    'bedrooms',
    'bathrooms',
    'beds',
    'price',
    'minimum_nights',
    'maximum_nights',
    'number_of_reviews']
# 导入数据
listing_data = pd.read_csv(filename)
# 截取需要的列
listing_data = listing_data[features]
# 查看数据集的行数和列数
print("数据集维度:")
print(listing_data.shape)
# 查看数据集前10行
peek = listing_data.head(10)
print("查看数据集前10行:")
print(peek)
# 查看每列的数据类型
print("查看每列的数据类型:")
print(listing_data.dtypes)
# 展示了八方面信息：数据记录数、平均值、标准方差、最小值、下四分位数、中位数、上四分位数、最大值
print("展示了八方面信息:数据记录数、平均值、标准方差(标准差)、最小值、下四分位数、中位数、上四分位数、最大值")
print(listing_data.describe())
# 数据属性的相关性
print("数据属性的相关性,皮尔逊相关系数:")
pd.set_option('display.width', 200)
print(listing_data.corr(method='pearson'))
# 数据分布分析
print("数据分布分析,高斯分布又叫正态分布:")
print(listing_data.skew())

# 直方图
listing_data.hist(bins=5)
plt.show()

# 密度图
listing_data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
plt.show()

# 箱线图
listing_data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False)
plt.show()

# 相关矩阵图
correlations = listing_data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, 9, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(features)
ax.set_yticklabels(features)
plt.show()

# 散点矩阵图
pd.plotting.scatter_matrix(listing_data)
plt.show()
