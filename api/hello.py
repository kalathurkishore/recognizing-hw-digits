from flask import Flask
from flask_restx import Resource, Api
from werkzeug.utils import cached_property

app = Flask(__name__)
api = Api(app)


@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


if __name__ == '__main__':
     app.run(host='0.0.0.0')
