# -*- coding: utf-8 -*-
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
	return jsonify({'message': 'Hello World'})

@app.route('/test', methods=['GET'])
def test():
	return jsonify({'test': 'test'})


def startAPI(host, port):
	app.run(debug=True, host=host, port=port) # remember to set debug to False
