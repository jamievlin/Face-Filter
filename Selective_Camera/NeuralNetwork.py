import numpy as np
import scipy
import scipy.optimize
import json
import io
import datetime
import random as rand
from PIL import Image

DEBUG_FLAG = False


class NeuralNetwork:
    def __init__(self, layers=None, lambda_val=0):
        self.layers = layers
        if self.layers is None:
            self.layers = [225, 25, 1]
        self.__rand_max__ = 0.15
        self.__params__ = [np.matrix(np.random.uniform(
            low=-self.__rand_max__, high=self.__rand_max__, size=(self.layers[i], self.layers[i-1] + 1)))
            for i in range(1, len(layers))]
        self.__lambda_val__ = lambda_val
        self.__train_data__ = np.matrix([])  # designer matrix form
        self.__train_label__ = np.matrix([])  # row vector form
        self.__data_vis__ = []
        self.__params_comments__ = ''

    def create_visualized_image(self, layer):
        if self.__data_vis__[layer] is not None:
            num_features, num_count = self.__data_vis__[layer].shape
            base_tiles_count = int(np.ceil(np.sqrt(num_count)))
            pic_res = int(np.ceil(np.sqrt(num_features)))
            vis_image = Image.new('L', (pic_res * base_tiles_count, pic_res * base_tiles_count))
            for result_index in range(num_count):
                new_array = self.__data_vis__[layer][:, result_index].getT()
                if new_array.shape[1] < (pic_res ** 2):
                    new_array = np.concatenate([new_array, [0.0] * ((pic_res ** 2) - new_array.shape[1])])
                img_array = np.reshape(new_array, (pic_res, pic_res))
                instance_image = Image.new('L', (pic_res, pic_res))
                for i in range(pic_res):
                    for j in range(pic_res):
                        img_array_val = int(round(img_array[i, j] * 255))
                        instance_image.putpixel((i, j), img_array_val)

                x_coord = (result_index % base_tiles_count) * pic_res
                y_coord = (result_index // base_tiles_count) * pic_res
                vis_image.paste(instance_image, (x_coord, y_coord))
            return vis_image

    def create_data_vis(self, num_count=-1, vis_data=None):
        if vis_data is None:
            vis_data = self.__train_data__
        m = vis_data.shape[0]
        if num_count > m:
            raise ValueError("Can't visualize more than the data count!")
        mat_shape = (num_count, self.layers[0])  # designer matrix
        if num_count == m or num_count < 0:
            num_count = m
            vis_sample = vis_data
        else:
            vis_sample = np.matrix(np.zeros(mat_shape))
            rand_sample = rand.sample(range(m), num_count)
            for rand_index in range(num_count):
                vis_sample[rand_index, :] = vis_data[rand_sample[rand_index], :]  # still in designer matrix form
        hyp, vis_list, *args = NeuralNetwork.__generic_hyp__(vis_sample.getT(), self.__params__)
        for a_result in vis_list:
            self.__data_vis__.append(a_result[1:, :])

    def load_data(self, train_data, train_label):
        self.__train_data__ = np.matrix(train_data)
        self.__train_label__ = np.matrix(train_label).getT()

    def save_param(self, location, includemetadata=True, comments=''):
        rolled_params = NeuralNetwork.roll_vec(self.__params__).tolist()
        out_object = {'layer_size': self.layers, 'rolled_params': rolled_params}

        if includemetadata:
            out_object['_metadata'] = {
            'date' : str(datetime.datetime.today()),
            'num_train_label': len(self.__train_data__),
            'cost_val': self.get_cost_train(NeuralNetwork.roll_vec(self.__params__)),
            'lambda_val': self.__lambda_val__
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

        self.load_param_json(json.loads(json_text))

    def load_param_json(self, json_obj):
        layer_data = json_obj['layer_size']

        assert (layer_data == self.layers), 'Layer data not equal!'

        rolled_param = np.array(json_obj['rolled_params'])
        self.__load_param_list__(rolled_param)

    def __load_param_list__(self, rolled_params):
        self.__params__ = self.unroll_params(rolled_params)

    @classmethod
    def sigmoid(cls, input_val):
        return 1/(1 + np.exp(-input_val))

    @classmethod
    def __get_generic_cost__(cls, param, train_data, train_label, lambda_val=0):
        assert lambda_val >= 0
        m = train_data.shape[0]
        raw_hyp, *args = NeuralNetwork.__generic_hyp__(train_data.getT(), param)
        hyp = raw_hyp.getT()
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

    @classmethod
    def __get_datapoint_grad__(cls, param, train_point, train_label):
        L = len(param) + 1

        total_grad = []

        # forward propagation
        hyp, a_data, *args = NeuralNetwork.__generic_hyp__(train_point, param)

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
            adjusted_delta = delta[l+1]
            if adjusted_delta.shape[0] > param[l].shape[0]:
                adjusted_delta = adjusted_delta[1:, :]
            delta[l] = np.multiply(param[l].getT() * adjusted_delta, temp_sigmoid_grad)

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
        return NeuralNetwork.roll_vec(NeuralNetwork.get_generic_grad(self.unroll_params(param), self.__train_data__,
                                                                     self.__train_label__, self.__lambda_val__))

    def get_cost_train(self, param):
        return NeuralNetwork.__get_generic_cost__(self.unroll_params(param), self.__train_data__, self.__train_label__,
                                                  self.__lambda_val__)

    def unroll_params(self, rolled_params):
        new_params = []
        curr_index = 0
        for theta in self.__params__:
            m, n = theta.shape
            new_param_temp = np.matrix(np.reshape(rolled_params[curr_index:curr_index + (m*n)], (m, n)))
            new_params.append(new_param_temp)
            curr_index = curr_index + (m * n)
        return new_params

    def train(self):
        print("Starting Training...")

        # test_val = self.__params__[0] - self.unroll_params(self.roll_vec(self.__params__))[0]

        rolled_inital_param = self.roll_vec(self.__params__)

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
        self.__params__ = self.unroll_params(np.array(rolled_new_parms))
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

    @classmethod
    def __generic_hyp__(cls, data, param):
        a = data
        a_list = []
        z_list = [np.array([])] # blank array. z_list[0] should never been accessed since a is the raw input data

        for theta in param:
            a_with_bias = np.concatenate([np.ones((1, a.shape[1])), a], axis=0)
            a_list.append(a_with_bias)
            z = theta * a_with_bias
            a = NeuralNetwork.sigmoid(z)
            z_list.append(z)
        a_list.append(np.concatenate([np.ones((1, a.shape[1])), a], axis=0))  # last one
        return a, a_list, z_list

    def hypothesis(self, data):
        parsed_data = np.matrix(data).getT()
        hyp, *args = NeuralNetwork.__generic_hyp__(parsed_data, self.__params__)
        return hyp
