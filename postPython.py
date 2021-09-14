import pickle
import numpy as np
from flask import Flask, jsonify, request, json

app = Flask(__name__)

# {"Temperature":0,"Pressure":0,"Humidity":0,"WindDirection(Degrees)":0,"Speed":0,"mes":0,"hora":0,"diferencia":0}

@app.route('/hola')
def hola():
    return 'Aqui'

@app.route('/prediction',methods=['POST'])
def prediction():
    content = request.get_json()

    datos = np.array([[content['Temperature'],content['Pressure'],content['Humidity'],content['WindDirection'],content['Speed'],content['mes'],content['hora'],content['diferencia']]])
    modelo = pickle.load(open("modelRF.h5", 'rb'))

    result = str(modelo.predict(datos)[0])
    print(datos)
    d = {"result": result}
    response = app.response_class(
        response=json.dumps(d),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == "__main__":
    app.run()
