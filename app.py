import numpy as np
import pandas as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')
    

@app.route("/predict",methods=['POST'])
def predict():
    features = np.array([x for x in request.form.values()])
    final_features = [np.array(features)]
    # input_df = pd.DataFrame(features.reshape(1,12),columns=['founded_at', 'first_funding_at', 'last_funding_at', 'funding_rounds',
    #    'funding_total_usd', 'first_milestone_at', 'last_milestone_at',
    #    'milestones', 'relationships', 'lat', 'lng', 'Active_Days])
    output = model.predict(final_features)
    print(output[0])
    if output[0] == 1:
        output = 'Closed'
    else:
        output = 'Not Closed'
    return render_template('index.html',prediction_test='The status of the company is : "{}"'.format(output))


# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)