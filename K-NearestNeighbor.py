import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#读取数据
pd.set_option('display.max_columns', None) #显示所有数据
#可容纳的旅客，卧室数量，厕所数量，床的数量，每晚费用，客人最少租几天，客人最多租几天，评论数量
features = ['accommodates', 'bedrooms', 'bathrooms', 'beds', 'price', 'minimum_nights', 'maximum_nights', 'number_of_reviews']
dc_listings = pd.read_csv('listings.csv')
dc_listings = dc_listings[features]
print(dc_listings.shape) #显示维度
print(dc_listings.head()) #输出前5行
print("=======================================")

#计算差异值
our_acc_value = 3
dc_listings['distance'] = np.abs(dc_listings.accommodates - our_acc_value)
#分别统计不同差异的数据条数
print(dc_listings.distance.value_counts().sort_index())
print("=======================================")

#数据洗牌
dc_listings = dc_listings.sample(frac=1, random_state=0)
dc_listings = dc_listings.sort_values('distance')
print(dc_listings.price.head())
print("=======================================")

#计算平均值
dc_listings['price'] = dc_listings.price.str.replace("\$|,", '').astype(float)
mean_price = dc_listings.price.iloc[:5].mean()
print(mean_price)
print("=======================================")

#制定训练集和测试集
dc_listings.drop('distance',axis=1)
train_df = dc_listings.copy().iloc[:2792]
test_df = dc_listings.copy().iloc[2792:]

#单变量预测价格
def predict_price(new_listing_value,feature_column):
    temp_df = train_df
    temp_df['distance'] = np.abs(dc_listings[feature_column] - new_listing_value)
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return (predicted_price)

test_df['predicted_price'] = test_df.accommodates.apply(predict_price,feature_column='accommodates')
print(test_df['predicted_price'])
print("=======================================")

#RMSE均方根误差
test_df['squared_error'] = (test_df['predicted_price'] - test_df['price']) ** (2)
mse = test_df['squared_error'].mean()
rmse = mse ** (1/2)
print(rmse)
print("=======================================")

#数据预处理
dc_listings = dc_listings.dropna()
print(dc_listings[features])
transformer = StandardScaler().fit(dc_listings[features])
dc_listings[features] = transformer.transform(dc_listings[features])
normalized_listing = dc_listings
print(dc_listings.shape)
print(normalized_listing.head())