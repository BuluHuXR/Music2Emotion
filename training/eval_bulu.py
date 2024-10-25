# coding: utf-8
from sklearn.preprocessing import LabelBinarizer
from torch.autograd import Variable
import model as Model
import numpy as np
import argparse
import torch
import tqdm
import os

MER31K_TAGS = ['genre---Confident', 'genre---Earnest', 'genre---Happy', 'genre---Passionate', 'genre---Playful', 'genre---Sad']

class Predict(object):
    def __init__(self, config):
        self.model_type = config.model_type
        self.model_load_path = config.model_load_path
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.is_cuda = torch.cuda.is_available()
        self.build_model()
        self.get_dataset()

    def get_model(self):
        if self.model_type == 'short':
            self.input_length = 59049
            return Model.ShortChunkCNN()
        elif self.model_type == 'short_res':
            self.input_length = 59049
            return Model.ShortChunkCNN_Res()
        else:
            print('model_type has to be one of [fcn, musicnn, crnn, sample, se, short, short_res, attention]')

    def build_model(self):
        self.model = self.get_model()

        # load model
        self.load(self.model_load_path)

        # cuda
        if self.is_cuda:
            self.model.cuda()

    def get_dataset(self):
        self.file_dict = np.load('./../split/MER31K/test_mer31k_nomo_test.npy', allow_pickle=True).item()
        self.test_list = list(self.file_dict.keys())
        self.mlb = LabelBinarizer().fit(MER31K_TAGS)


    def load(self, filename):
        S = torch.load(filename)
        if 'spec.mel_scale.fb' in S.keys():
            self.model.spec.mel_scale.fb = S['spec.mel_scale.fb']
        self.model.load_state_dict(S)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def get_tensor(self, fn):
        # load audio
        filename = self.file_dict[fn]['path']
        npy_path = os.path.join(self.data_path, "npy", filename)

        raw = np.load(npy_path, mmap_mode='r')

        # ## predict
        length = len(raw)
        hop = length // 2 - self.input_length // 2

        x = torch.zeros(1, self.input_length)
        x[0] = torch.Tensor(raw[hop: hop + self.input_length]).unsqueeze(0)
        #
        return x

    def test(self):
        self.model = self.model.eval()
        est_array = []
        for line in tqdm.tqdm(self.test_list):
            fn = line

            # load and split
            x = self.get_tensor(fn)

            # forward
            x = self.to_var(x)
            out = self.model(x)
            out = out.detach().cpu()

            # estimate
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)

            # # # #
            predicted_labels = MER31K_TAGS[np.argmax(estimated)]
            print("predicted_labels", predicted_labels)

            pred_labels = np.argmax(estimated)

            em_prediction_onehot = torch.zeros(6)
            em_prediction_onehot[pred_labels] = 1
            print("em_prediction_onehot", em_prediction_onehot)

if __name__ == '__main__':
    import time

    start = time.time()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_type', type=str, default='short', choices=['short', 'short_res'])
    parser.add_argument('--model_load_path', type=str, default='./../models/MER31K/short/best_model.pth')
    parser.add_argument('--data_path', type=str, default='./nomo_song_wav_2')
    parser.add_argument('--batch_size', type=int, default=16)
    config = parser.parse_args()

    p = Predict(config)
    p.test()

    print(f"total used time：{time.time() - start}")
