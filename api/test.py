import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
import joblib
import numpy as np


# Nên Scaler để chuẩn hóa dữ liệu.

# twitter_data = pd.read_csv("./dataset/dataset.csv")

# twitter_data = twitter_data.drop(['tweetFollower', 'tweetFollowing', 'tweetId'], axis=1)
# twitter_data['targetY'] = 3 * twitter_data['replyReply'] + 3 * twitter_data['replyRetweet'] + 2 * twitter_data['replyLike'] + \
#                           twitter_data['replyView'] + twitter_data['webAccess'] + 2 * twitter_data['webComment'] + \
#                           3 * twitter_data['webRate'] + 4 * twitter_data['webOrder']

# X = twitter_data.drop(['replyReply','replyRetweet','replyLike','replyView','webAccess','webComment','webOrder','webRate','targetY'], axis=1)
# Y = twitter_data['targetY']

# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

# regressor = RandomForestRegressor(n_estimators=100)
# regressor.fit(X_train,Y_train)

# joblib.dump(regressor, './models/model-ai.pkl')

# test_data_prediction = regressor.predict(X_test)
# error_score = sklearn.metrics.r2_score(Y_test, test_data_prediction)
# print("R squared error : ", error_score)
# print(test_data_prediction)

model = joblib.load('./models/model-ai.pkl')


input_data = [12,135,731,19360,1326,9946,45829,9498914,0,0,0]
input_array = np.array(input_data)
input_array_reshaped = input_array.reshape(1,-1)

print(input_array_reshaped)

prediction = model.predict(input_array_reshaped)
print(prediction)






