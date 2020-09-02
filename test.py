import torch
import os
import numpy as np
from datasets.tree import get_test_dataset
from models.vgg import vgg19
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import argparse

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--image', type=str, required=True,
                        help='image filename')
    parser.add_argument('--tree_pts', type=str, required=True,
                        help='projected tree points')
    parser.add_argument('--data_split', type=str, required=True,
                        help='file containing info about data split')
    parser.add_argument('--save-dir', default='/home/teddy/vgg',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = get_test_dataset(args.image, args.tree_pts, args.data_split)
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))
    epoch_minus = []

    def save_images(path,count,inputs,outputs):
        inputs = inputs.cpu().numpy()
        outputs = outputs.cpu().numpy()
        inputs = np.transpose(inputs[0,:3],(1,2,0))
        outputs = np.squeeze(outputs)
        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(inputs)
        plt.title('%d'%count)
        plt.subplot(1,2,2)
        plt.imshow(outputs)
        plt.title('%f'%np.sum(outputs))
        plt.savefig(path)

    i = 0
    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            #save_images('image%04d.png'%i,count[0].item(),inputs,outputs)
            i += 1
            temp_minu = count[0].item() - torch.sum(outputs).item()
            print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
            epoch_minus.append(temp_minu)

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
