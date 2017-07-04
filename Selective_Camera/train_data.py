import NeuralNetwork as nn
import PIL.Image
import numpy as np

def main():
    save_location = input('Where to save parameters (JSON): ')
    img_res_side = 25
    test_nn = nn.NeuralNetwork([img_res_side ** 2, 25, 1])
    test_nn.load_data(create_data(img_res_side, './data/test_data/Kim_Jong-Il.jpg'), 0)

    test_nn.save_param(save_location, includemetadata=True, comments='Test parameter. Not trained yet.')
    print('Success. Saved to', save_location)


def create_data(img_res_side, img_file):
    test_img = PIL.Image.open(img_file)
    test_img = test_img.resize((img_res_side, img_res_side)).convert('L')
    test_list = []
    img_1_feature = []
    for x in range(0, img_res_side):
        for y in range(0, img_res_side):
            img_1_feature.append(test_img.getpixel((x, y)) / 255)
    test_list.append(img_1_feature)
    return test_list

if __name__ == '__main__':
    main()
