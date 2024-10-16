import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify

def predict_price(input_data, model, scaler):
    try:
        df_input2 = pd.DataFrame([input_data])
        X = df_input2 
        X = scaler.transform(X)
        y_prediction = model.predict(X)
        return y_prediction[0]
    except Exception as e:
        return str(e)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def index():
    if request.method == 'POST':
        try:
            boarding = str(request.form.get('from'))
            destination = str(request.form.get('Destination'))
            selected_flight_class = str(request.form.get('flightType'))
            selected_agency = str(request.form.get('agency'))
            week_no = int(request.form.get('week_no'))
            week_day = int(request.form.get('week_day'))
            day = int(request.form.get('day'))

            boarding = 'from_' + boarding
            boarding_city_list = ['from_Florianopolis (SC)', 'from_Sao_Paulo (SP)', 'from_Salvador (BH)', 'from_Brasilia (DF)', 'from_Rio_de_Janeiro (RJ)', 'from_Campo_Grande (MS)', 'from_Aracaju (SE)', 'from_Natal (RN)', 'from_Recife (PE)']

            destination = 'destination_' + destination
            destination_city_list = ['destination_Florianopolis (SC)', 'destination_Sao_Paulo (SP)', 'destination_Salvador (BH)', 'destination_Brasilia (DF)', 'destination_Rio_de_Janeiro (RJ)', 'destination_Campo_Grande (MS)', 'destination_Aracaju (SE)', 'destination_Natal (RN)', 'destination_Recife (PE)']

            selected_flight_class = 'flightType_' + selected_flight_class
            class_list = ['flightType_economic', 'flightType_firstClass', 'flightType_premium']

            selected_agency = 'agency_' + selected_agency
            agency_list = ['agency_Rainbow', 'agency_CloudFy', 'agency_FlyingDrops']

            travel_dict = dict()

            for city in boarding_city_list:
                travel_dict[city] = 1 if city[:-5] == boarding else 0
            for city in destination_city_list:
                travel_dict[city] = 1 if city[:-5] == destination else 0
            for flight_class in class_list:
                travel_dict[flight_class] = 1 if flight_class == selected_flight_class else 0
            for agency in agency_list:
                travel_dict[agency] = 1 if agency == selected_agency else 0
            travel_dict['week_no'] = week_no
            travel_dict['week_day'] = week_day
            travel_dict['day'] = day

            scaler_model = pickle.load(open("model/scaling.pkl", 'rb'))
            rf_model = pickle.load(open("model/rf_model.pkl", 'rb'))

            predicted_price = str(round(predict_price(travel_dict, rf_model, scaler_model), 2))
            return jsonify({'prediction': predicted_price})

        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
