from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
import os
import json
import sklearn.metrics

model_filename = os.path.join(os.path.dirname(__file__), 'models/model-ai.pkl')
csv_filename = os.path.join(os.path.dirname(__file__), 'dataset/dataset.csv')

@api_view(["GET"])
def buildModel(request):
        twitter_data = pd.read_csv(csv_filename)
        twitter_data = twitter_data.drop(['tweetId'], axis=1)
        twitter_data['targetY'] = 3 * twitter_data['replyReply'] + 3 * twitter_data['replyRetweet'] + 2 * twitter_data['replyLike'] + \
                                twitter_data['replyView'] + twitter_data['webAccess'] + 2 * twitter_data['webComment'] + 2 * twitter_data['webUser']  + \
                                3 * twitter_data['webRate'] + 4 * twitter_data['webOrder']
        X = twitter_data.drop(['replyReply','replyRetweet','replyLike','replyView','webAccess','webComment','webOrder','webUser','webRate','targetY'], axis=1)
        Y = twitter_data['targetY']
        X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=2)
        regressor = RandomForestRegressor(n_estimators=100)
        regressor.fit(X_train,Y_train)

        # R sqaured error
        test_data_prediction = regressor.predict(X_test)
        error_score = sklearn.metrics.r2_score(Y_test, test_data_prediction)
        print("R squared error : ", error_score)

        # save model
        joblib.dump(regressor, model_filename)
        return Response({"message":"build model successfully","error_score":error_score}) 


@api_view(["POST"])
def prediction(request):
    body = json.loads(request.body)
    input_data = body["data"]

    input_array = np.array(input_data)
    input_array_reshaped = input_array.reshape(1,-1)

    model = joblib.load(model_filename)
    prediction = model.predict(input_array_reshaped)
    return Response({"data":prediction[0]})    