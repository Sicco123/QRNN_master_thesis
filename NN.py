import numpy as np

class qrnn(object):
    def __init__(self):
        self.n_layers = 0
        self.neuron_list = []
        self.weight_list = []
        self.bias_list = []
        self.activation_list = []

    def add_layer(self, n_neurons, input_dim = None, activation = self.sigmoid):
        """
        :param n_neurons: Number of neurons of the new layer
        :param input_dim: Dimension of the in input of the new layer
        :param activation: The activation function
        :return: The neural network with an extra layer
        """

        if input_dim is not None: # The first time a layer is added the input_dim should be specified
            self.n_input = input_dim
        else:
            self.n_input = self.neuron_list[-1]

        self.neuron_list.append(self.n_neurons)

        weights_and_bias = self.initialize_weights_and_bias(self.n_input, n_neurons) # Random initialisation of weights and biases
        self.weight_list.append(weights_and_bias[:-1,:])
        self.bias_list.append(weights_and_bias[-1,:])

        self.activation_list.append(activation) # set the activation of the layer
        self.n_layer += 1

    def initialize_weights_and_bias(self, n_input, n_neurons):
        weights = np.random.uniform(low=0.0, high=1.0, size = (n_neurons,n_input+1))
        return weights

    def sigmoid(x):
        sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
        return sig

    def forward(self, observation):
        input = observation
        for layer in range(self.n_layer):
            inter_output =  input.T @ self.weight_list[layer] + self.bias_list[layer]
            output = self.activation_list[layer](inter_output)
            input = output

        return output

    def fit(self, input, output, l_rate, n_epoch):
        for epoch in range(n_epoch):
            for t in range(len(input[:,0])):
                observation = input[t,:]

