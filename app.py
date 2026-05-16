from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    fever = int(request.form['fever'])
    cough = int(request.form['cough'])
    headache = int(request.form['headache'])
    fatigue = int(request.form['fatigue'])

    features = np.array([[fever, cough, headache, fatigue]])

    prediction = model.predict(features)

    return render_template(
        'result.html',
        disease=prediction[0]
    )

if __name__ == '__main__':
    app.run(debug=True)