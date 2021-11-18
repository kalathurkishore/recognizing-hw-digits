from flask import Flask
from flask import request
from flask_restx import Resource, Api
from werkzeug.utils import cached_property
from joblib import dump, load
import numpy as np

best_model_path = '../models/tt_0.15_val_0.15_rescale_1_gamma_0.001/model.joblib'

app = Flask(__name__)
#api = Api(app)

clf = load(best_model_path)

'''@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}
'''

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict",methods=['POST'])
def predict():
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1,-1)
    prediction = clf.predict(image)
    output = '\n Predicted digit is : ' + str(prediction[0]) + '\n\n'
    return  output
    #return "<p>image received</p>"



'''
if __name__ == '__main__':
     app.run(debug=True)'''
