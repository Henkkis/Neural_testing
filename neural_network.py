


import numpy as np
import subprocess


class NeuronLayer:
    def __init__(self,number_of_neurons,inputs_per_neuron,momentum =0.9):
        self.synaptic_weights = 2*np.random.random((inputs_per_neuron,number_of_neurons))-1
        self.synaptic_delta = 0
        self.synaptic_delta_previous =0
        self.momentum = momentum
        self.bias=2*np.random.random((1,number_of_neurons))
        self.bias_delta = 0


class MlpNetwork:
    
    # network_layout is a vector with the number of neurons at each level
    def __init__(self,network_layout,learning_rate = 1,sigmoid_parameter = 1, output_function = "sigmoid" ):
            self.layers = []
            self.values = np.array ([ i*[None] for i in network_layout ])
            self.lr = learning_rate
            self.b  = sigmoid_parameter

            
            if(output_function == "sigmoid"):
                self.output_function = self.__sigmoid
                self.output_error = self.__sigmoid_error
            
            elif(output_function == "linear"):
                print("Activated linear output function")
                self.output_function = self.__linear
                self.output_error = self.__linear_error
            else:
                self.output_function = NONE
                self.outout_error = NONE

            
            assert sigmoid_parameter > 0
            assert learning_rate > 0

            for i in range(0,len(network_layout)-1):
                self.layers.append( NeuronLayer(network_layout[i+1],network_layout[i]))
    
    def __sigmoid(self,x):
        return 1/(1+np.exp(-self.b*x))

    def __linear(self,x):
        return x
    
    def __linear_error(self,x):
        return 1

    def __sigmoid_error(self,x):
        return x*(1-x)

    # progagates the inputs through the network ie. evaluates the input
    def propagate(self,inputs):
        current_state = inputs
        self.values[0] = current_state
        value_idx = 1

        # Propagate through the hidden layers
        for layer in self.layers[0:-1]:
            current_state = self.__sigmoid(np.dot(current_state,layer.synaptic_weights)+layer.bias)    
            self.values[value_idx] = current_state
            current_state = current_state 
            value_idx+=1
        # Propagate through the output
        current_state = self.output_function(np.dot(current_state,self.layers[-1].synaptic_weights) +self.layers[-1].bias  )    
        self.values[-1] = current_state
        return current_state

    # Updates the weights in the network using backwards propagation
    def __calc_wdelta(self,targets):
        node_values = reversed(self.values)
        nvalue = next(node_values)
        output_error = (targets-nvalue)*self.output_error(nvalue)
        nvalue = next(node_values)
        wdelta =  self.lr*np.dot(nvalue.T,output_error)
        self.layers[-1].bias_delta = output_error*self.lr
        prev_weight =  np.copy( self.layers[-1].synaptic_weights) 
        self.layers[-1].synaptic_delta = wdelta        
        prev_error = output_error
        for layer in reversed(self.layers[0:-1]):
            curr_error = nvalue*(1-nvalue)*np.dot(prev_error,prev_weight.T)
            layer.bias_delta += curr_error*self.lr
            nvalue = next(node_values)
            wdelta =  self.lr*np.dot(nvalue.T,curr_error)
            prev_weight = np.copy(layer.synaptic_weights)
            layer.synaptic_delta = wdelta 
            prev_error =  curr_error



    # Trains the network
    def train(self,input_data,target_data,num_iterations):
        assert len(input_data) == len(target_data)
        for iteration in range(0,num_iterations):
            idx = np.random.randint(len(target_data))
            self.propagate(input_data[np.newaxis,idx])
            self.__calc_wdelta(target_data[np.newaxis,idx])
            for layer in self.layers:
                layer.synaptic_weights += layer.synaptic_delta + layer.momentum*layer.synaptic_delta_previous
                layer.synaptic_delta_previous = layer.synaptic_delta
                layer.synaptic_delta =0
                layer.bias += layer.bias_delta
                layer.bias_delta = 0
 
    def draw_network(self):

        len_array = [max(i.shape) for i in self.values]
        num_inputs = len_array[0]
        num_outputs = len_array[-1]
        num_hidden_layers = len(len_array)-2
        Nco_array = ';'.join( list(map(str,len_array))[1:])
        struct_array = ','.join(list(map(str,len_array))[1:-1])
        

        param_string = ("\n\def\innum{" + str(num_inputs) +"}"
         +"\n\def\outnum{" +str(num_outputs)+"}"
         +"\n\def\\numhidden{" +str(num_hidden_layers) + "}"
         +"\n\def\\networkstruct{" +struct_array+"}"
         +"\n\setarray{Nco}{" + Nco_array + "}\n")
        fh = open("neural_params.tex","w")
        fh.write(param_string)
        fh.close()
        subprocess.run(["pdflatex","-interaction=batchmode", "neural.tex"])


#random testing
#a = MlpNetwork([1,3,3,1],0.2,1,"linear")
#x=np.ones((1,40))*np.linspace(0,1,40)
#t=np.sin(2*np.pi*x) + np.cos(4*np.pi*x)
#x=x.T
#t=t.T

#train = x[0::2,:]
#test = x[1::4,:]
#valid = x[3::4,:]
#traintarget = t[0::2,:]
#testtarget = t[1::4,:]
#validtarget = t[3::4,:]
#
#inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
#outputs = np.array([[0, 1, 1, 1, 1, 0, 0]]).T
#a.train(train,traintarget,60000)
#pl.plot(train,traintarget,'.')
#pl.plot(test,a.propagate(test),'.')
#pl.show()
#print (a.propagate(np.array([[1,1,0]])))
#print (a.propagate(np.array([[0,0,1]])))
#print (a.propagate(np.array([[0,1,1]])))
#print (a.propagate(np.array([[1,0,1]])))
#print (a.propagate(np.array([[0,1,0]])))
#print (a.propagate(np.array([[1,0,0]])))
#print (a.propagate(np.array([[1,1,1]])))
#print (a.propagate(np.array([[0,0,0]])))

#print a.layers[0].synaptic_weights
#print a.layers[1].synaptic_weights


#    a.draw_network()

