from pyforest import *
import joblib
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from flask_cors import cross_origin

app = Flask(__name__)

# loading DS operations and files

# load models
#model_catboost = joblib.load("models/CatBoostRegressor.pkl")
model_xgboost = joblib.load("models/XGBoostRegressor.pkl")


@app.route("/")
@cross_origin()
def home():
    return render_template("flask.html")


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    # feature extraction
    #features = [x for x in request.form.values()]

    flight_date = datetime.strptime(request.form['Flight Date'], '%Y-%m-%d')
    creation_date_time = datetime.strptime(
        request.form['Booking date'], '%Y-%m-%d')
    flight_number = request.form['Flight Number']
    board_point = request.form['Origin']
    off_point = request.form['Destination']
    fare_level_code = request.form['Fare Level Code']

    # dataframe creation
    df = pd.DataFrame()
    df['FLIGHT_DATE'] = [flight_date]
    df['CREATION_DATE_TIME'] = [creation_date_time]
    df['FLIGHT_NUMBER'] = [flight_number]
    df['BOARD_POINT'] = [board_point]
    df['OFF_POINT'] = [off_point]
    df['FARE_LEVEL_CODE'] = [fare_level_code]

    # data transformation
    df['FLIGHT_DATE'] = pd.to_datetime(df['FLIGHT_DATE'])
    df['CREATION_DATE_TIME'] = pd.to_datetime(df['CREATION_DATE_TIME'])
    df['DAYS_GAP'] = (df['FLIGHT_DATE'] - df['CREATION_DATE_TIME'])
    df['DAYS_GAP'] = df['DAYS_GAP'].astype('timedelta64[D]')
    df['FLIGHT_DAY_OF_WEEK'] = df['FLIGHT_DATE'].dt.weekday
    df['FLIGHT_MONTH'] = df['FLIGHT_DATE'].dt.month
    df['FLIGHT_YEAR'] = df['FLIGHT_DATE'].dt.year
    df['BOOKING_DAY_OF_WEEK'] = df['CREATION_DATE_TIME'].dt.weekday
    df['BOOKING_MONTH'] = df['CREATION_DATE_TIME'].dt.month
    df['BOOKING_YEAR'] = df['CREATION_DATE_TIME'].dt.year

    # model
    df_final = df[['FLIGHT_NUMBER', 'BOARD_POINT', 'OFF_POINT', 'FARE_LEVEL_CODE', 'DAYS_GAP',
                   'FLIGHT_DAY_OF_WEEK', 'FLIGHT_MONTH', 'FLIGHT_YEAR', 'BOOKING_DAY_OF_WEEK', 'BOOKING_MONTH', 'BOOKING_YEAR']]

    result = model_xgboost.predict(df_final)

    output = round(result[0])
    return render_template('flask.html', prediction_text='Your prediction : {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
