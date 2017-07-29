import numpy as np
import scipy
import scipy.optimize
import json
import io
import datetime

DEBUG_FLAG = False


class NeuralNetwork:
    def __init__(self, layers=None, lambda_val=0):
        self.layers = layers
        if self.layers is None:
            self.layers = [225, 25, 1]
        self.rand_max = 0.15
        self.params = [np.matrix(np.random.uniform(
            low=-self.rand_max, high=self.rand_max, size=(self.layers[i], self.layers[i-1] + 1)))
            for i in range(1, len(layers))]
        self.lambda_val = lambda_val
        self.train_data = np.matrix([])  # designer matrix form
        self.train_label = np.matrix([])  # row vector form

    @classmethod
    def parse_data(cls, data):
        return np.matrix(data).getT()

    def load_data(self, train_data, train_label):
        self.train_data = np.matrix(train_data)
        self.train_label = np.matrix(train_label).getT()

    def save_param(self, location, includemetadata=True, comments=''):
        rolled_params = NeuralNetwork.roll_vec(self.params).tolist()
        out_object = {'layer_size': self.layers, 'rolled_params': rolled_params}

        if includemetadata:
            out_object['_metadata'] = {
            'date' : str(datetime.datetime.today()),
            'num_train_label': len(self.train_data),
            'cost_val': self.get_cost_train(NeuralNetwork.roll_vec(self.params)),
            'lambda_val': self.lambda_val
            }
        if comments:
            out_object['_comments'] = comments

        out_text = json.dumps(out_object, indent=True)
        file = io.open(location, 'w')
        file.write(out_text)
        file.close()

    def load_param(self, file_location):
        param_file = io.open(file_location, 'r')
        json_text = param_file.read()
        param_file.close()

        self.load_param_json(json_text)

    def load_param_json(self, json_params_str):
        param_object = json.loads(json_params_str)
        layer_data = param_object['layer_size']

        assert (layer_data == self.layers), 'Layer data not equal!'

        rolled_param = np.array(param_object['rolled_params'])
        self.load_param_str(rolled_param)

    def load_param_str(self, rolled_params):
        self.params = self.unroll_params(rolled_params)

    @classmethod
    def sigmoid(cls, input_val):
        return 1/(1 + np.exp(-input_val))

    @classmethod
    def __get_generic_cost__(cls, param, train_data, train_label, lambda_val=0):
        assert lambda_val >= 0
        m = train_data.shape[0]
        hyp = NeuralNetwork.__generic_hyp__(train_data.getT(), param).getT()
        hyp_log = np.log(hyp)
        cost_comp_1 = np.multiply(-train_label, hyp_log)
        cost_comp_2 = np.multiply(-(1 - train_label), np.log(1 - hyp))
        cost = (1/m) * np.sum(cost_comp_1 + cost_comp_2)

        reg_cost = 0
        if lambda_val > 0:
            for theta in param:
                penalty_mat = np.power(np.delete(theta, 0, axis=1), 2)
                reg_cost = reg_cost + np.sum(penalty_mat)
            reg_cost = (lambda_val / (2 * m)) * reg_cost

        return cost + reg_cost

    # TODO: Check Gradient Error. (shouldn't have error within 0.35 range)
    @classmethod
    def __get_datapoint_grad__(cls, param, train_point, train_label):
        L = len(param) + 1

        # forward propagation
        a_data = [np.matrix([0])] * L
        a_data[0] = np.concatenate([np.matrix([1]), train_point], axis=0)

        total_grad = []
        for i in range(1, L):
            temp_z = param[i-1] * a_data[i-1]
            unbiased_a = np.matrix(NeuralNetwork.sigmoid(temp_z))
            a_data[i] = np.concatenate([np.matrix([1]), unbiased_a], axis=0)

        # back propagation
        delta = [np.matrix([0])] * L
        delta[0] = None  # shouldn't be used.
        delta[L-1] = a_data[L-1][1:, :] - train_label

        # [L-2 -> 1] inclusive. (or layer L-1 -> 2 starting from 1)
        # calculating delta for backpropagation
        for l in range(L-2, 0, -1):  # excluding first layer
            assert l > 0
            raw_a = a_data[l]
            temp_sigmoid_grad = np.multiply(raw_a, (1 - raw_a))
            delta[l] = np.multiply(param[l].getT() * delta[l+1], temp_sigmoid_grad)

        for theta_layer in range(0, L-1):
            sliced_delta = delta[theta_layer + 1]
            if theta_layer != L-2:
                sliced_delta = sliced_delta[1:, :]
            total_grad.append(np.matmul(sliced_delta, a_data[theta_layer].getT()))
        return total_grad

    @classmethod
    def get_generic_grad(cls, param, train_data, train_label, lambda_val=0):
        assert lambda_val >= 0, 'Lambda cannot be less than zero!'
        l = len(param)
        m = train_data.shape[0]  # still in designer matrix form.
        temp_grad = [np.zeros(param[i].shape) for i in range(l)]
        total_grad = [np.zeros(param[i].shape) for i in range(l)]

        for data_index in range(m):
            data_grad = NeuralNetwork.__get_datapoint_grad__(param, train_data[data_index, :].getT(),
                                                             train_label[data_index])
            for theta_layer in range(l):
                temp_grad[theta_layer] = temp_grad[theta_layer] + data_grad[theta_layer]

        for theta_layer in range(0, l):
            if lambda_val > 0:
                penalty_mat = np.power(np.matrix(param[theta_layer]), 2)
                penalty_mat[:, 0] = np.zeros([penalty_mat.shape[0], 1])
                total_grad[theta_layer] = lambda_val * penalty_mat
            total_grad[theta_layer] = total_grad[theta_layer] + ((1/m) * temp_grad[theta_layer])

        return total_grad

    @classmethod
    def roll_vec(cls, mat_list):
        result = []
        for mat in mat_list:
            result_list = mat.flatten('C')
            result.extend(result_list.tolist()[0])
        return np.array(result)

    def get_grad(self, param):
        return NeuralNetwork.roll_vec(NeuralNetwork.get_generic_grad(self.unroll_params(param), self.train_data,
                                                                     self.train_label, self.lambda_val))

    def get_cost_train(self, param):
        return NeuralNetwork.__get_generic_cost__(self.unroll_params(param), self.train_data, self.train_label,
                                                  self.lambda_val)

    def unroll_params(self, rolled_params):
        new_params = []
        curr_index = 0
        for theta in self.params:
            m, n = theta.shape
            new_param_temp = np.matrix(np.reshape(rolled_params[curr_index:curr_index + (m*n)], (m, n)))
            new_params.append(new_param_temp)
            curr_index = curr_index + (m * n)
        return new_params

    #@classmethod
    #def generic_unroll_params(cls, rolled_params, param_template):
    #    new_params = []
    #    curr_index = 0
    #    for theta in param_template:
    #        (m,n) = theta.shape()
    #        new_param_temp = np.matrix(np.reshape(rolled_params[curr_index:curr_index + (m*n)], (m,n)))
    #        new_params.append(new_param_temp)
    #        curr_index = curr_index + (m * n)
    #    return new_params

    def train(self):
        print("Starting Training...")

        # test_val = self.params[0] - self.unroll_params(self.roll_vec(self.params))[0]

        rolled_inital_param = self.roll_vec(self.params)

        print("Initial Cost/Grad:", self.get_cost_train(rolled_inital_param), self.get_grad(rolled_inital_param))
        print("Rolled Parameters length:", str(rolled_inital_param.size))

        if DEBUG_FLAG:
            error = scipy.optimize.check_grad(self.get_cost_train, self.get_grad, rolled_inital_param, epsilon=0.0001)
            print('Grad Error', error)

        use_gradient = True
        if use_gradient:
            f_grad = self.get_grad
        else:
            f_grad = False
        rolled_new_parms = scipy.optimize.minimize(fun=self.get_cost_train, x0=rolled_inital_param, method='CG',
                                                   jac=f_grad, callback=self.optim_callback, options={
                                                       'disp': True
                                                   })['x']
        self.params = self.unroll_params(np.array(rolled_new_parms))
        print("Training Finished!")

    def optim_callback(self, xk):
        temp_params = xk
        print("Iteration success.")
        print("XK Parameter:", str(xk))
        print("Current Cost: ", self.get_cost_train(temp_params))
        if DEBUG_FLAG:
            error = scipy.optimize.check_grad(self.get_cost_train, self.get_grad, temp_params, epsilon=0.0001)
            print('Grad Error', error)
        print('-'*25)
    #@classmethod
    #def generic_train(cls, nn_inst):
    #    assert isinstance(nn_inst, NeuralNetwork)
    #    new_params = scipy.optimize.fmin_bfgs(lambda param : NeuralNetwork.__get_generic_cost__(NeuralNetwork.generic_unroll_params(param, nn_inst.params), nn_inst.train_data, nn_inst.train_label, nn_inst.lambda_val), x0=NeuralNetwork.roll_vec(nn_inst.params), fprime = lambda param : NeuralNetwork.roll_vec(NeuralNetwork.get_generic_grad(NeuralNetwork.generic_unroll_params(param, nn_inst.params), nn_inst.train_data, nn_inst.train_label, nn_inst.lambda_val)) )
    #    print("Training success!")
    #    return new_params

    @classmethod
    def __generic_hyp__(cls, data, param):
        a = data
        for theta in param:
            a_with_bias = np.concatenate([np.ones((1, a.shape[1])), a], axis=0)
            a = NeuralNetwork.sigmoid(theta * a_with_bias)
        return a

    def hypothesis(self, data):
        return NeuralNetwork.__generic_hyp__(data, self.params)
