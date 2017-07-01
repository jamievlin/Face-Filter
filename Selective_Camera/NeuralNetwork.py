import numpy as np
import scipy
import scipy.optimize
import json, io

class NeuralNetwork:
    def __init__(self,layers = [225, 25, 1], lambda_val = 0):
        self.layers = layers
        self.rand_max = 0.001;
        self.params = [np.matrix((np.random.rand(layers[i], layers[i-1] + 1) * (2 * self.rand_max)) - self.rand_max) for i in range(1, len(layers))]
        self.lambda_val = lambda_val
        self.train_data = [] # designer matrix form
        self.test_label = [] # row vector form


    def load_data(self,train_data, train_label):
        self.train_data = np.matrix(train_data)
        self.train_label = np.transpose(np.matrix(train_label))

    def save_param(self, location):
        rolled_params = NeuralNetwork.roll_vec(self.params).tolist()
        out_object = {'layer_size':self.layers, 'rolled_params':rolled_params} 
        out_text = json.dumps(out_object, indent=True)
        file = io.open(location, 'w')
        file.write(out_text)
        file.close()

    @classmethod
    def sigmoid(cls, input):
        return 1/(1 + np.exp(-input))

    @classmethod
    def __get_generic_cost__(cls, param, train_data, train_label, lambda_val = 0):
        m = train_data.shape[0]
        hyp = NeuralNetwork.__generic_hyp__(train_data.getT(), param).getT()
        hyp_log = np.log(hyp)
        cost = -(1/m) * (np.multiply(train_label,hyp_log) + np.multiply((1 - train_label), np.log(1 - hyp)))

        reg_cost = 0;
        if lambda_val > 0:
            for theta in param:
                penalty_mat = np.power(np.delete(theta, (0), axis=1),2)
                reg_cost = reg_cost + np.sum(penalty_mat)
            reg_cost = (lambda_val / ( 2 * m)) * reg_cost

        return float(cost + reg_cost)

    @classmethod
    def __get_datapoint_grad__(cls, param, train_point, train_label):
        L = len(param) + 1
        a_data = [np.array([0])] * L
        a_data[0] = np.concatenate([np.ones((1, train_point.shape[1])), train_point],axis=0)
        delta = [np.array([0])] * L
        total_grad = []
        for i in range(1, L):
            temp_z = param[i-1] * a_data[i-1]
            unbiased_a = NeuralNetwork.sigmoid(temp_z)
            a_data[i] = np.concatenate([np.ones((1, unbiased_a.shape[1])), unbiased_a],axis=0)

        delta[L-1] = a_data[L-1] - train_label

        for l in range(L-2, 0, -1):
            temp_sigmoid_grad = np.multiply(a_data[l], (1 - a_data[l]))
            delta[l] = np.multiply(param[l].getT() * delta[l+1][1:,0:], temp_sigmoid_grad)

        for theta_layer in range(0, L-1):
            total_grad.append(delta[theta_layer + 1][1:, 0:] * a_data[theta_layer].getT())
        return total_grad

    @classmethod
    def get_generic_grad(cls, param, train_data, train_label, lambda_val = 0):
        L = len(param) + 1
        m = train_data.shape[0]
        total_grad = [np.zeros(param[i].shape) for i in range(0, L-1)]

        for data_index in range(0, m):
            temp_grad = NeuralNetwork.__get_datapoint_grad__(param, np.matrix(train_data[data_index, 0:]).getT(), train_label[data_index])
            for theta_layer in range(0, L-1):
                total_grad[theta_layer] = total_grad[theta_layer] + (1/m) * temp_grad[theta_layer]
        return total_grad

    @classmethod
    def roll_vec(cls, mat_list):
        result = []
        for mat in mat_list:
            result_list = mat.flatten('F')
            result.extend(result_list.tolist()[0])
        return np.array(result)

    def get_grad(self, param):
        return NeuralNetwork.roll_vec(NeuralNetwork.get_generic_grad(self.unroll_params(param), self.train_data, self.train_label, self.lambda_val))

    def get_cost_train(self, param):
        return NeuralNetwork.__get_generic_cost__(self.unroll_params(param), self.train_data, self.train_label, self.lambda_val)

    def unroll_params(self, rolled_params):
        new_params = []
        curr_index = 0
        for theta in self.params:
            m, n = theta.shape
            new_param_temp = np.matrix(np.reshape(rolled_params[curr_index:curr_index + (m*n)], (m,n)))
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
        rolled_inital_param = self.roll_vec(self.params)
        train_func = lambda param : self.get_cost_train(param)
        grad_func = lambda param : self.get_grad(param)

        print("Initial Cost/Grad:", self.get_cost_train(rolled_inital_param), self.get_grad(rolled_inital_param))
        print("Rolled Parameters length:", str(rolled_inital_param.size))

        rolled_new_parms = scipy.optimize.fmin_bfgs(f=train_func, x0=rolled_inital_param, fprime=grad_func,disp=True, retall=True, callback = self.optim_callback)
        self.params  = self.unroll_params(np.array(rolled_new_parms[0]))
        print("Training Finished!")

    def optim_callback(self, xk):
        temp_params = xk
        print("Iteration success.")
        print("XK Parameter:", str(xk))
        print("Current Cost: ", self.get_cost_train(temp_params))

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
            a_with_bias = np.concatenate([
                np.ones((1, a.shape[1])),
                a
                ], axis=0)
            a = NeuralNetwork.sigmoid(theta * a_with_bias)
        return np.matrix(a)

    def hypothesis(self, data):
        return NeuralNetwork.__generic_hyp__(data, self.params)
