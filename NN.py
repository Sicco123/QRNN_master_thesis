import numpy as np
import math
class qrnn(object):
    def __init__(self):
        self.n_layers = 0
        self.neuron_list = []
        self.weight_list = []
        self.bias_list = []
        self.activation_list = []
        self.inter_output_list = []
        self.inter_input_list = []

    def add_layer(self, n_neurons, input_dim = None, activation = self.tanh):
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

        weights, bias = self.initialize_weights_and_bias(self.n_input, n_neurons) # Random initialisation of weights and biases
        self.weight_list.append(weights)
        self.bias_list.append(bias)

        self.activation_list.append(activation) # set the activation of the layer
        self.n_layer += 1

    def initialize_weights_and_bias(self, n_input, n_neurons):
        weights = np.random.uniform(low=0.0, high=1.0, size = (n_neurons,n_input))
        bias = np.ramdon.uniform()
        return weights, bias

    def hramp(self, x):
        if np.size(self.lower) >1:
            return map(self.hramp, x, self.lower, self.eps)
        else :
            if math.isinf(self.lower) and self.lower < 0 :
                return x
            else:
                return (self.huber(x-self.lower,self.eps) if x > self.lower else 0) + self.lower

    def hramp_prime(self, x):
        if np.size(self.lower) > 1 :
             return map(self.hramp_prime, x, self.lower, self.eps)
        else :
            if math.isinf(self.lower) and self.lower < 0 :
                return 1
            else:
                dhr = (x - self.lower)/self.eps
                dhr[x > (self.lower + self.eps)] = 1
                dhr[x < self.lower] = 0
        return dhr


    def huber(self, x):
        h = np.abs(x) - self.eps/2 if abs(x) > self.eps else (x**2) / (2 * self.eps)
        h[np.where(np.isnan(h))] = 0
        return h

    def huber_prime(self, x):
        dh = x / self.eps
        dh[np.where(x > self.eps)] = 1
        dh[np.where(x < -self.eps)] = -1
        dh[np.isnan(dh)] = 0
        return dh

    def tilted_approx(self, x):
        res = self.tau* self.huber(x,self.eps) if x > 0 else (1-self.tau)*self.huber(x,self.eps)
        return res

    def tilted_approx_prime(self, x):
        res = self.tau*self.huber_prime(x,self.eps) if x > 0 else (1-self.tau)*self.huber_prime(x,self.eps)
        return res

    def tanh(self, x):
        tanh_res = np.tanh(x)
        return tanh_res

    def tanh_prime(self, x):
        tanh_prime_res = 1 - np.tanh(x)**2
        return tanh_prime_res
    def sigmoid(self, x):
        sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
        return sig

    def sigmoid_prime(self, x):
        sig_prime = self.sigmoid(x) * (1-self.sigmoid(x))
        return sig_prime

    def forward(self, observations):
        input = observations
        for layer in range(self.n_layer):
            self.inter_input_list.append(input)
            inter_output = input @ self.weight_list[layer] + self.bias_list[layer]
            self.inter_output_list.append(inter_output)
            output = self.activation_list[layer](inter_output)
            input = output

        return output

    def get_derivative(self, layer):
        if self.activation_list[layer] == self.hramp:
            derivative = self.hramp_prime
        elif self.activation_list[layer] == self.tanh:
            derivative = self.tanh_prime
        elif self.activation_list[layer] == self.sigmoid:
            derivative = self.sigmoid_prime

        return derivative

    def get_delta(self, layer, errors):
        activation_function = self.activation_list[layer]
        inter_output = self.inter_output_list[layer]

        if activation_function == self.hramp:
            delta = self.hramp_prime(inter_output,self.lower, self.eps)*self.tilted_approx_prime(self, errors)
        elif activation_function == self.tanh:
            delta = self.tanh_prime(inter_output)*errors
        elif activation_function == self.sigmoid:
            delta = self.sigmoid_prime(inter_output)*errors

        return delta

    def backward(self, errors):
        self.gradient_list = []

        for idx, layer in enumerate(reversed(range(self.n_layer))):
            weights = self.weight_list[layer]
            bias = self.bias_list[layer]
            #derivative = self.get_derivative(layer)
            delta = self.get_delta(layer, errors)
            gradient_weights = -(np.append(self.inter_input_list[layer],1).T@(delta*self.w)) #1/len(errors) are the weights, these could be changed
            gradient_weights[:-1, ] = gradient_weights[:-1,:] * weights # only when monotonicity is required
            gradient_penalty = 0 if idx == 0 else 2*self.penalty*np.append(weights,bias)/(len(weights)+1-len(weights[0]))
            errors = delta @ weights.T

            self.graident_list.append((gradient_weights + gradient_penalty))

    def cost_func(self, errors):
        cost = self.w*np.sum(self.tilted_approx(self,errors))

        for idx, layer in enumerate(reversed(range(self.n_layer))):
            penalty_component = 0 if idx ==0 else self.penalty*(self.weight_list[layer]**2)/np.sum(self.weight_list[layer])
            cost += penalty_component

        return cost

    def fit(self, input, output, n_epoch, lower, eps, tau, penalty):
        T = len(input[:,0])
        self.lower = lower
        self.eps = eps
        self.tau = tau
        self.penalty = penalty
        self.w = 1/len(input)
        for epoch in range(n_epoch):
            estimates = self.forward(input)
            errors = output - estimates
            self.backward(errors)
            cost = self.cost_func(errors)
            gradient = self.gr