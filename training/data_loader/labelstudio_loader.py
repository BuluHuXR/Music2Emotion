# coding: utf-8
from sklearn.preprocessing import LabelBinarizer
from torch.utils import data
import numpy as np
import csv
import os

META_PATH = './../split/labelstudio/'

labelstudio_TAGS = ['genre---IT', 'genre---HQ', 'genre---PB', 'genre---LE', 'genre---CF']

def read_file(tsv_file):
    tracks = {}
    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            track_id = row[0]
            tracks[track_id] = {
                'path': row[3].replace('.wav', '.npy'),
                'tags': row[5:],
            }
    return tracks

class AudioFolder(data.Dataset):
    def __init__(self, root, split, input_length=None):
        self.root = root
        self.split = split
        self.input_length = input_length
        self.get_songlist()

    def __getitem__(self, index):
        npy, tag_binary = self.get_npy(index)
        return npy.astype('float32'), tag_binary.astype('float32')

    def get_songlist(self):
        self.mlb = LabelBinarizer().fit(labelstudio_TAGS)
        if self.split == 'TRAIN':
            train_file = os.path.join(META_PATH, 'jtrain.npy')
            self.file_dict = np.load(train_file, allow_pickle=True).item()
            self.fl = list(self.file_dict.keys())
        elif self.split == 'VALID':
            train_file = os.path.join(META_PATH,'jvalid.npy')
            self.file_dict= np.load(train_file, allow_pickle=True).item()
            self.fl = list(self.file_dict.keys())
        elif self.split == 'TEST':
            test_file = os.path.join(META_PATH, 'jtest.npy')
            self.file_dict= np.load(test_file, allow_pickle=True).item()
            self.fl = list(self.file_dict.keys())
        else:
            print('Split should be one of [TRAIN, VALID, TEST]')

    def get_npy(self, index):
        jmid = self.fl[index]
        filename = self.file_dict[jmid]['path']
        npy_path = os.path.join(self.root, "npy", filename)
        npy = np.load(npy_path, allow_pickle=True)#, mmap_mode='r'
        random_idx = int(np.floor(np.random.random(1) * (len(npy)-self.input_length)))
        npy = np.array(npy[random_idx:random_idx+self.input_length])
        tag_binary = np.sum(self.mlb.transform(self.file_dict[jmid]['tags']), axis=0)
        return npy, tag_binary

    def __len__(self):
        return len(self.fl)

def get_audio_loader(root, batch_size, split='TRAIN', num_workers=0, input_length=None):
    data_loader = data.DataLoader(dataset=AudioFolder(root, split=split, input_length=input_length),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader