'''

NeuralNet class creates a Q-learining network by binding together different neural network layers

'''

import os, scipy
from ConvolutionalLayer import *
from HiddenLayer import *
from OutputLayer import *

theano.config.openmp = False  # they say that using openmp becomes efficient only with "very large scale convolution"
theano.config.floatX = 'float32'

class NeuralNet:

    def __init__(self, input_shape, filter_shapes, strides, n_hidden, n_out):
        '''
        Initialize a NeuralNet

        @param input_shape: tuple or list of length 4 , (batch size, num input feature maps,
                             image height, image width)
        @param filter_shapes: list of 2 (for each conv layer) * 4 values (number of filters, num input feature maps,
                              filter height,filter width)
        @param strides: list of size 2, stride values for each hidden layer
        @param n_hidden: int, number of neurons in the all-to-all connected hidden layer
        @param n_out: int, number od nudes in output layer
        '''

        #create theano variables corresponding to input_batch (x) and output of the network (y)
        x = T.ftensor4('x')
        y = T.fmatrix('y')

        #first hidden layer is convolutional:
        self.layer_hidden_conv1 = ConvolutionalLayer(x, filter_shapes[0], input_shape, strides[0])

        #second convolutional hidden layer: the size of input depends on the size of output from first layer
        #it is defined as (num_batches, num_input_feature_maps, height_of_input_maps, width_of_input_maps)
        second_conv_input_shape = [input_shape[0], filter_shapes[0][0], self.layer_hidden_conv1.feature_map_size,
                                   self.layer_hidden_conv1.feature_map_size]
        self.layer_hidden_conv2 = ConvolutionalLayer(self.layer_hidden_conv1.output, filter_shapes[1],
                                                     image_shape=second_conv_input_shape, stride=2) # Drops use of strides

        #output from convolutional layer is 4D, but normal hidden layer expects 2D. Because of all to all connections
        # 3rd hidden layer does not care from which feature map or from which position the input comes from
        flattened_input = self.layer_hidden_conv2.output.flatten(2)

        #create third hidden layer
        self.layer_hidden3 = HiddenLayer(flattened_input, self.layer_hidden_conv2.fan_out, n_hidden)

        #create output layer
        self.layer_output = OutputLayer(self.layer_hidden3.output, n_hidden, n_out)

        #define the ensemble of parameters of the whole network
        self.params = self.layer_hidden_conv1.params + self.layer_hidden_conv2.params \
            + self.layer_hidden3.params + self.layer_output.params

        #discount factor
        self.gamma = 0.95

        #: define regularization terms, for some reason we only take in count the weights, not biases)
        #  linear regularization term, useful for having many weights zero
        self.l1 = abs(self.layer_hidden_conv1.W).sum() \
            + abs(self.layer_hidden_conv2.W).sum() \
            + abs(self.layer_hidden3.W).sum() \
            + abs(self.layer_output.W).sum()

        #: square regularization term, useful for forcing small weights
        self.l2_sqr = (self.layer_hidden_conv1.W ** 2).sum() \
            + (self.layer_hidden_conv2.W ** 2).sum() \
            + (self.layer_hidden3.W ** 2).sum() \
            + (self.layer_output.W ** 2).sum()

        #: define the cost function
        self.cost = 0.0 * self.l1 + 0.0 * self.l2_sqr + self.layer_output.errors(y)
        self.cost_function = theano.function([x, y], [self.cost])

        #: define gradient calculation
        self.grads = T.grad(self.cost, self.params)

        #: Define how much we need to change the parameter values
        self.learning_rate = T.scalar('lr')
        self.updates = []
        for param_i, gparam_i in zip(self.params, self.grads):
            self.updates.append((param_i, param_i - self.learning_rate * gparam_i))
        self.x = x
        self.y = y

        #: we need another set of theano variables (other than x and y) to use in train and predict functions
        temp_x = T.ftensor4('temp_x')
        temp_y = T.fmatrix('temp_y')

        #: define the training operation as applying the updates calculated given temp_x and temp_y
        self.train_model = theano.function(inputs=[temp_x, temp_y,
                                                   theano.Param(self.learning_rate, default=0.00001)],
                                           outputs=[self.cost],
                                           updates=self.updates,
                                           givens={
                                               x: temp_x,
                                               y: temp_y},
                                           name='train_model')

        self.cost_clone = theano.clone(self.cost, replace=self.updates)
        self.line_function = theano.function(
            [x, y, self.learning_rate], [self.cost_clone])

        self.predict_rewards = theano.function(
            inputs=[temp_x],
            outputs=[self.layer_output.output],
            givens={
                x: temp_x
            },
            name='predict_rewards')

        self.predict_rewards_and_cost = theano.function(
            inputs=[temp_x, temp_y],
            outputs=[self.layer_output.output, self.cost],
            givens={
                x: temp_x,
                y: temp_y
            },
            name='predict_rewards_and_cost')

    actual_learning_rate = 1e-5
    learning_rates = []

    def optimal_learning_rate(self, prestates, new_estimated_Q, lr):
        objective = lambda lr: self.line_function(np.array(prestates), new_estimated_Q, float(lr))[0]
        res = scipy.optimize.minimize(objective, 0, method='Nelder-Mead', options={'xtol': 1e-1})
        print 'optimization result'
        print res
        self.learning_rates.append(float(res.x))

    def train(self, minibatch):
        """
        Train function that transforms (state,action,reward,state) into (input, expected_output) for neural net
        and trains the network
        @param minibatch: array of dictionaries, each dictionary contains
        one transition (prestate,action,reward,poststate)
        """
        prestates = [t['prestate'] for t in minibatch]
        initial_estimated_Q = self.predict_rewards(prestates)[0]
        new_estimated_Q = initial_estimated_Q.copy()
        post_eQ = self.predict_rewards([t['poststate'] for t in minibatch])[0]
        actions = [t['action'] for t in minibatch]
        rewards = np.array([t['reward'] for t in minibatch])
        for row, (peQ, action, reward) in enumerate(zip(post_eQ, actions, rewards)):
            new_estimated_Q[row, action] = reward + self.gamma * np.max(peQ)
        initial_cost = self.cost_function(prestates, new_estimated_Q)
        optimal_learning_rate = lambda: self.optimal_learning_rate(
            prestates, new_estimated_Q,
            self.learning_rates[-1] if self.learning_rates else self.actual_learning_rate)
        if (len(self.learning_rates) % 50) == 0:
            print 'computing optimal learning rate'
            optimal_learning_rate()
        else:
            self.learning_rates.append(self.learning_rates[-1])
        self.train_model(np.array(prestates), new_estimated_Q,
                         self.learning_rates[-1])
        final_cost = self.cost_function(prestates, new_estimated_Q)
        final_estimated_Q = self.predict_rewards(prestates)[0]
        print 'initial_cost', initial_cost, 'final_cost', final_cost, 'foo baz'
        print 'current rewards', (final_estimated_Q - final_estimated_Q.min(axis=0)).mean(axis=0)
        print 'current rewards absolute'
        for r, a, s in sorted(zip(rewards, actions, map(list, final_estimated_Q))):
            print r, a, s
        if final_cost > initial_cost:
            print 'overstepped; computing current optimal learning rate'
            optimal_learning_rate()
        if os.path.exists('/var/tmp/stop'):
            import pdb
            pdb.set_trace()

    def predict_best_action(self, state):
        """
        Predict_best_action returns the action with the highest Q-value
        @param state: 4D array, input (game state) for which we want to know the best action
        """
        predicted_values_for_actions = self.predict_rewards(state)[0][0]
        return  np.argmax(predicted_values_for_actions)