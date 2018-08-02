import numpy as np



class HopNetwork:
    
    def __init__(number_of_neurons):
        self.weights = np.random.random((number_of_neurons,number_of_neurons))
        np.fill_diagonal(self.weights,0)
        self.neurons = 2*np.random.random((1,number_of_neurons))-1
        self.weights = np.array(map(lambda x: ,self.neurons))
    
    def train(input_data):
        

