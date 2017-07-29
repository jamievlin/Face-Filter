#!/usr/bin/env python3

import NeuralNetwork as nn
import PIL.Image
import os
import os.path
import json
import io
import datetime
import math

DEBUG_ENABLED = False


def main():
    usr_valid_responses = {'T', 'H'}
    usr_response = ''
    while usr_response not in usr_valid_responses:
        usr_response = input('(H)ypothesis or (T)rain? ')

    if usr_response == 'T':
        img_res_side = 25
        train_nn = nn.NeuralNetwork([img_res_side ** 2, 25, 1])
        train_data(train_nn, img_res_side, output_path='data/trained_params')
    elif usr_response == 'H':
        hypothesis_main()


def hypothesis_main():
    default_params_dir = 'data/trained_params/'
    file_list = get_file_list(default_params_dir, '.json')
    print('Parameters found: ')
    for file_index in range(len(file_list)):
        print(str.format('{0}. {1}', str(file_index + 1), file_list[file_index]))
    usr_response = None

    while not usr_response:
        usr_response = input('Enter a number or another parameter (from the main directory): ')
        if usr_response.isnumeric():
            int_response = int(usr_response)
            if 1 <= int_response <= len(file_list):
                usr_response = os.path.join(default_params_dir, file_list[int_response - 1])
            else:
                usr_response = None
                print('Error. Index out of range.')

    param_obj = load_json_obj(usr_response)
    test_nn = nn.NeuralNetwork(param_obj['layer_size'])
    test_nn.load_param_json(param_obj)

    image_res = int(math.sqrt(param_obj['layer_size'][0]))
    hyp_data, *args, img_list = create_test_data('data/test_data/', img_res=image_res)
    hypothesis(test_nn, hyp_data, img_list)


def load_json_obj(file_name):
    json_file = io.open(file_name)
    param_str = json_file.read()
    json_file.close()
    return json.loads(param_str)


def get_file_list(directory, extension):
    return [name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))
            and name.endswith(extension)]


def hypothesis(input_neural_network, data, face_file_list=None):
    assert isinstance(input_neural_network, nn.NeuralNetwork)
    result = input_neural_network.hypothesis(nn.NeuralNetwork.parse_data(data)).tolist()[0]

    for i in range(len(result)):
        result_hyp = result[i]
        if face_file_list is not None:
            print('Face data: ', face_file_list[i])
        print('Raw Hypothesis: ', str(result_hyp))
        if result_hyp >= 0.5:
            print('PASSED: Well, it seems I like this face. :)')
        else:
            print('FAILED: Well, looks like not this one :(')
        print()
        print('-'*20)
        print()


def train_data(input_neural_network, img_res_data, output_path=None, output_name=None):
    assert isinstance(input_neural_network, nn.NeuralNetwork)

    path_folder = 'data/train_data/'
    label_path = 'data/train_label.json'

    label_file = io.open(label_path)
    json_str = label_file.read()
    label_file.close()

    label_obj = json.loads(json_str)

    data, label, *args = create_test_data(path_folder, label_obj, img_res_data)
    input_neural_network.load_data(data, label)
    print(str.format('Input size: {0}', str(len(data))))
    print('Starting training.')
    input_neural_network.train()
    print('Training successful.')

    if output_name is None:
        output_name = str.format('traindata_{0}.json',
                                 str(datetime.datetime.now()).replace(':', '-').replace(' ', '_'))
        # for compatibility issues (Windows doesn't accept ':' in filename)
    if output_path is None:
        output_path = ''  # empty string

    input_neural_network.save_param(os.path.join(output_path, output_name))


def create_test_data(path, label=None, img_res=25):
    img_extensions = {'.jpg', '.png', '.tiff', '.tif'}

    image_list = get_file_list(path, tuple(img_extensions))

    final_data = []
    final_label = []
    for image_file in image_list:
        file_path = os.path.join(path, image_file)
        if (label is None) or (image_file in label and not image_file.startswith('_')):
            final_data.append(create_data(img_res, file_path))
        if label is not None and image_file in label:
            final_label.append(float(label[image_file]))

        if DEBUG_ENABLED:
            print('Appended', file_path)

    assert len(final_data) == len(final_label) or label is None
    return final_data, final_label, image_list


def create_data(img_res_side, img_file):
    test_img = PIL.Image.open(img_file)
    test_img = test_img.resize((img_res_side, img_res_side)).convert('L')
    img_1_feature = []
    for x in range(0,img_res_side):
        for y in range(0, img_res_side):
            img_1_feature.append(test_img.getpixel((x, y)) / 255)
    return img_1_feature

if __name__ == '__main__':
    main()
