from flask import Flask,request,jsonify
from libs.model import multnomial_nb , logixtic_regression


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/api/accuracy_and_predicted_score/' , methods = ['POST'])
def predict_val():
    path_= request.json['path_']
    model = request.json['model']
    if (model == 'Logistic Regression'):

        accuracy , predicted_score = logixtic_regression(path=path_)

    elif(model == 'Naive Bayes'):
        accuracy , predicted_score = multnomial_nb(path=path_)

    else:
        print('error')
    return jsonify({'model':model ,'accuracy':accuracy , 'predicted_class':predicted_score})
    


if __name__ =="__main__" :
    app.run(host='0.0.0.0' ,debug=True  , port = 8200)

