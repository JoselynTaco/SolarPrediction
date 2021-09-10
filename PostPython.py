import pickle
import numpy as np
from sklearn import preprocessing
from flask import Flask, jsonify, request, json

def abrirModelo(ruta):
    modelo = pickle.load(open(ruta, 'rb'))
    return modelo

def predecir(modelo, X):
    return modelo.predict(X)

app = Flask(__name__)

# {"Temperature":0,"Pressure":0,"Humidity":0,"WindDirection(Degrees)":0,"Speed":0,"mes":0,"hora":0,"diferencia":0}

@app.route('/hola')
def holamundo():
    return 'Aqui'

@app.route('/prueba',methods=['POST'])
def prueba():
    content = request.get_json()

    Temperature = content['Temperature']
    Pressure = content['Pressure']
    Humidity = content['Humidity']
    WindDirection = content['WindDirection']
    Speed = content['Speed']
    mes = content['mes']
    hora = content['hora']
    diferencia = content['diferencia']

    datos = np.array([[Temperature,Pressure,Humidity,WindDirection,Speed,mes,hora,diferencia]])
    datos = preprocessing.scale(datos)

    result = str(predecir(abrirModelo("modelRF.h5"), datos)[0])
    d = {"result": result}
    response = app.response_class(
        response=json.dumps(d),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == "__main__":
    app.run()