from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
import os
import json

model_filename = os.path.join(os.path.dirname(__file__), 'models/model-ai.pkl')
csv_filename = os.path.join(os.path.dirname(__file__), 'dataset/dataset.csv')

@api_view(["GET"])
def buildModel(request):
        twitter_data = pd.read_csv(csv_filename)
        twitter_data = twitter_data.drop(['tweetFollower', 'tweetFollowing', 'tweetId'], axis=1)
        twitter_data['targetY'] = 3 * twitter_data['replyReply'] + 3 * twitter_data['replyRetweet'] + 2 * twitter_data['replyLike'] + \
                                twitter_data['replyView'] + twitter_data['webAccess'] + 2 * twitter_data['webComment'] + \
                                3 * twitter_data['webRate'] + 4 * twitter_data['webOrder']
        X = twitter_data.drop(['replyReply','replyRetweet','replyLike','replyView','webAccess','webComment','webOrder','webRate','targetY'], axis=1)
        Y = twitter_data['targetY']
        X_train,_,Y_train,_ = train_test_split(X,Y,test_size=0.2,random_state=2)
        regressor = RandomForestRegressor(n_estimators=100)
        regressor.fit(X_train,Y_train)

        # save model
        joblib.dump(regressor, model_filename)
        return Response({"message":"build model successfully"}) 


@api_view(["POST"])
def prediction(request):
    body = json.loads(request.body)
    input_data = body["data"]

    input_array = np.array(input_data)
    input_array_reshaped = input_array.reshape(1,-1)

    model = joblib.load(model_filename)
    prediction = model.predict(input_array_reshaped)
    return Response({"data":prediction})    