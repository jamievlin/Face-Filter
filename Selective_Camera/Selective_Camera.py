import NeuralNetwork as nn
import PIL.Image
import numpy as np


def main():
    img_res_side = 25
    test_nn = nn.NeuralNetwork([img_res_side ** 2, 25, 1])
    # test_nn.load_data(create_data(img_res_side), [1])
    # cost = test_nn.get_cost_train(nn.NeuralNetwork.roll_vec(test_nn.params))
    # test_nn.train()
    test_nn.load_param('output1.json')
    data = nn.NeuralNetwork.parse_data(create_data(img_res_side, './data/test_data/Kim_Jong-Il.jpg'))
    result = test_nn.hypothesis(data).tolist()[0][0]
    print('Face data: ', './data/test_data/Kim_Jong-Il.jpg')
    print('Raw Hypothesis: ', str(result))

    if result >= 0.5:
        print('PASSED: Well, it seems I like this face. :)')
    else:
        print('FAILED: Well, looks like not this one :(')

    print(result)


def create_data(img_res_side, img_file):
    test_img = PIL.Image.open(img_file)
    test_img = test_img.resize((img_res_side, img_res_side)).convert('L')
    test_list = []
    img_1_feature = []
    for x in range(0,img_res_side):
        for y in range(0, img_res_side):
            img_1_feature.append(test_img.getpixel((x, y)) / 255)
    test_list.append(img_1_feature)
    return test_list

if __name__ == '__main__':
    main()
