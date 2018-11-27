import neural_network  
import numpy as np


# Edit this
dimensions = [3,4,5,3]


# Dont edit this
A = neural_network.MlpNetwork(dimensions)
A.propagate(np.random.random((1,dimensions[0]) )  )
A.draw_network()
