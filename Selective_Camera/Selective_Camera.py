import NeuralNetwork as nn
import PIL.Image

def main():
    img_res_side = 25
    test_nn = nn.NeuralNetwork([img_res_side ** 2, 25, 1])
    test_nn.load_data(create_data(img_res_side), [1])
    # cost = test_nn.get_cost_train(nn.NeuralNetwork.roll_vec(test_nn.params))
    #test_nn.train()
    test_nn.train()
    test_nn.save_param('/output1.json')
    pass

def create_data(img_res_side):
    test_img = PIL.Image.open('data/raw_data/atty.png')
    test_img = test_img.resize((img_res_side,img_res_side)).convert('L')
    test_list = []
    img_1_feature = []
    for x in range(0,img_res_side):
        for y in range(0,img_res_side):
            img_1_feature.append(test_img.getpixel((x,y)) / 255)
    test_list.append(img_1_feature)
    return test_list

if __name__ == '__main__':
    main()
