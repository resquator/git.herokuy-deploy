from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def show_predict_rent_form():
    return render_template('predictorform.html')


@app.route('/results', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
        model_file_path = 'rent_price_predictor.pkl'        
        model = pickle.load(open(model_file_path, 'rb'))
        row_one = np.array([131,   2,   4,   0,   1]).reshape(1,-1) 
        predicted_rent_price = model.predict(row_one)
        return render_template('resultsform.html', prediction=np.round(predicted_rent_price[0],2))

if __name__ == '__main__':
    #app.run(debug=True)
    app.run()
