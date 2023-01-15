import os
import cv2
import numpy as np
import mindspore as ms
import argparse
from att_resnet import att_resnet50
from resnet import resnet50

num_class = 54

def imread(im_path, shape=None, color="RGB", mode=cv2.IMREAD_UNCHANGED):
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    if color == "RGB":
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if shape != None:
        assert isinstance(shape, int) 
    im = cv2.resize(im, (shape, shape))
    return im


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--test-path", type=str, default="test_images")
    parser.add_argument("--attention", type=bool, default=None) #None还是False?
    args = parser.parse_args()

    if args.attention:
        net = att_resnet50(num_class)
        param_dict = ms.load_checkpoint('weights/Att_4layers_best.ckpt')
    else:
        net = resnet50(num_class)
        param_dict = ms.load_checkpoint('weights/DA_resnet50_best.ckpt')

    test_path = args.test_path
    ms.load_param_into_net(net, param_dict)
    model = ms.Model(net)
    filenames = os.listdir(test_path)
    with open("result.txt","w") as f:
        for filename in filenames:
            img_path = os.path.join(test_path,filename)
            img = imread(img_path, shape=224, color='RGB')
            img = img.astype(np.float32)/255 # Rescale(1.0 / 255.0, 0.0)
            img = (img - [0.4914, 0.4822, 0.4465]) / [0.2023, 0.1994, 0.2010] # Normalize
            img = np.transpose(img, (2, 0, 1)) # HWC2CHW
            img = img.reshape(1,3,224,224)
            img = ms.Tensor(img,dtype=ms.float32)
            output = model.predict(img)
            pred = np.argmax(output.asnumpy(), axis=1)
            f.write(filename+' '+str(pred[0])+'\n')
    mp = dict()
    with open("result.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            k = int(line.split('.')[0])
            v = line.split(' ')[1]
            mp[k] = v
        mp = sorted(mp.items())
    with open("result.txt","w") as f:
        for item in mp:
            f.write(item[1])
