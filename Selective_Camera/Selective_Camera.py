import NeuralNetwork as nn
import PIL.Image
import os
import os.path
import json
import io
import datetime

DEBUG_ENABLED = False


def main():
    img_res_side = 25
    test_nn = nn.NeuralNetwork([img_res_side ** 2, 25, 1])
    usr_valid_responses = {'T', 'H'}
    usr_response = ''
    while usr_response not in usr_valid_responses:
        usr_response = input('(H)ypothesis or (T)rain? ')

    if usr_response == 'T':
        train_data(test_nn, img_res_side)
    elif usr_response == 'H':
        param_file = input('Load params? (Leave blank for default): ')
        if param_file:
            test_nn.load_param(param_file)
        hyp_data, *args, img_list = create_test_data('data/test_data/', img_res=img_res_side)
        hypothesis(test_nn, hyp_data, img_list)


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


def train_data(input_neural_network, img_res_data, output_path=None):
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

    if output_path is None:
        output_path = str.format('traindata_{0}.json', str(datetime.datetime.now()))

    input_neural_network.save_param(output_path)


def create_test_data(path, label=None, img_res=25):
    img_extensions = {'.jpg', '.png', '.tiff', '.tif'}
    cond = lambda path_dir, file_name: os.path.isfile(os.path.join(path_dir, file_name)) and file_name.endswith(tuple(img_extensions))
    image_list = [name for name in os.listdir(path) if cond(path, name)]

    final_data = []
    final_label = []
    for image_file in image_list:
        file_path = os.path.join(path, image_file)
        final_data.append(create_data(img_res, file_path))
        if label is not None:
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
