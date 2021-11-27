from flask import Flask
from flask import request
from flask_restx import Resource, Api
from werkzeug.utils import cached_property
from joblib import dump, load
import numpy as np

best_model_path_1 = '../models/tt_0.15_val_0.15_rescale_1_gamma_0.001/model.joblib'
best_model_path_2 = '../models/tt_0.1_val_0.1_rescale_1_depth_35/model.joblib'
app = Flask(__name__)
#api = Api(app)

clf = load(best_model_path_1)
clf1 = load(best_model_path_2)

'''@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}
'''

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict_svm",methods=['POST'])
def predict_svm():
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1,-1)
    prediction = clf.predict(image)
    output = '\n Predicted digit is : ' + str(prediction[0]) + '\n\n'
    return  output
    #return "<p>image received</p>"

@app.route("/predict_dt",methods=['POST'])
def predict_dt():
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1,-1)
    prediction = clf1.predict(image)
    output = '\n Predicted digit is : ' + str(prediction[0]) + '\n\n'
    return  output
    #return "<p>image received</p>"




if __name__ == '__main__':
     app.run(debug=True,host='0.0.0.0')
