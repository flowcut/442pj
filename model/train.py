from .eccv16 import *
import torch.optim as optim
import tqdm
from torch.utils.data import Dataset, DataLoader
import json
import glob
from .util import *
from skimage import color
import os


class ImageDataset(Dataset):

    def __init__(self, transform=None):
        with open('config.json') as config_file:
            config = json.load(config_file)
        dataset_dir = config["image_dir"]
        self.image_list = glob.glob(dataset_dir)
        color_map_dir = config["color_map"]
        self.index2color = {}
        self.color2index = {}
        with open(color_map_dir, 'r') as current_file:
            for line in current_file:
                index, row, col = line.rstrip('\n').split(',')
                self.index2color[index] = (row, col)
                self.color2index[(row, col)] = index
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.image_list[idx]
        original_image = resize_img(load_img(image_name))
        if self.transform is not None:
            original_image = self.transform(original_image)
        original_image = color.rgb2lab(original_image)
        img_l_rs = original_image[:, :, 0]
        img_ab_rs = original_image[:, :, 1:]
        mapped_img = np.zeros((original_image.shape[0],
                               original_image.shape[1]))
        for i in range(original_image.shape[0]):
            for j in range(original_image.shape[1]):
                temp = tuple((img_ab_rs[i][j] // 10) + 11)
                mapped_img[i][j] = self.color2index[temp]
        gaussian_image = 0  # TODO
        image = {
            'grey': img_l_rs,
            'mapped': mapped_img,
            'gaus': gaussian_image
        }

        return image


def train(model, criterion, dataloader, config, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_traj = []

    for epoch in tqdm.tqdm_notebook(range(epochs)):

        loss_epoch = 0
        for index, item in enumerate(dataloader):
            grey_x = item['grey']
            map_x = item['mapped']
            gaus_x = item['gaus']
            optimizer.zero_grad()
            pred = model(grey_x)
            loss = criterion(map_x, gaus_x, pred)
            loss.backward()
            optimizer.step()

            loss_traj.append(loss_epoch)

            if (index + 1) % config['save_step'] == 0:
                print('Epoch {}, loss {:.3f}'.format(epoch, loss_epoch))
                torch.save(model.state_dict(),
                           os.path.join(config['model_dir'],
                                        'model-{}-{}.ckpt'.format(epoch + 1,
                                                                  index + 1)))
