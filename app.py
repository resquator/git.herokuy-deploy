from flask import Flask, render_template, request
import pickle
import numpy as np
import logging

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
        
        surface = float(request.form['surface'])
        level = float(request.form['level'])
        room = float(request.form['room'])
        garden = float(request.form['garden'])
        garage = float(request.form['garage']) 
        
        print(" ")
        print("----------------------------------------------------------")
        print(f"Surface = {surface} of type {type(surface)}")
        row_one = np.array([surface,   level,   room,   garden,   garage]).reshape(1,-1) 
        #row_one.astype(np.float64)  
        
        logging.info(f"call the model with surface {surface}, level {level} ")

        predicted_rent_price = model.predict(row_one)
        return render_template('resultsform.html', prediction=np.round(predicted_rent_price[0],2))

if __name__ == '__main__':
    #app.run(debug=True)
    app.run()
