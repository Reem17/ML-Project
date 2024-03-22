from flask import Flask, request, render_template
from Inference import get_data, preprocess, Model, predict

app = Flask(__name__, template_folder='templates')
model = Model()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def inference():  
    response=request.form
    d = get_data(response)
    x = preprocess(d)
    pred = predict(model, x)
    
    output=""
    if pred == "M":
        output= "Tumor Diagnosis is Malignant (Positive)"
    else:
        output= "Tumor Diagnosis is Benign (Negative)"
            
    return render_template('result.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True,threaded=True)
