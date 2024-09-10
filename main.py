import numpy as np

# Funciones de activacion

def Relu(x):
    return max(0,x)

#def SoftMax():

def perceptron(entradas, pesos, sesgo):
    sumatoria = 0
    for i in range(len(entradas)):
        sumatoria += entradas[i] * pesos[i]
    sumatoria += sesgo
    return Relu(sumatoria)
        

def individuoNueronal():
    return 0;


print(Relu(80))

# test perceptron
o = perceptron([0.5,5,12,84],[1,2,6,87],1)

print(o)
