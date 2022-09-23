from torch.utils.data import Dataset
from REMGeneration.utils import get_terrain_from_info
import os
import pandas as pd
import numpy as np
import json

class REMStaticDataset(Dataset):

    def __init__(self,path,transforms,reference_rem_path=None):

        self.path = path
        self.rem_path = os.path.join(path,'rems')

        with open(os.path.join(path,'terrain_info.json'),'r') as f:
            self.terrain_info = json.load(f)

        self.mapping = pd.read_csv(os.path.join(path,'rem_mapping.csv'))
        self.mapping = self.mapping.set_index('Id')

        self.n_terrains = len(os.listdir(self.rem_path))
        self.transforms = transforms
        self.reference_path = reference_rem_path

        if reference_rem_path:
            self.reference_path = reference_rem_path
            self.reference_rem_path = os.path.join(reference_rem_path,'rems')
            self.reference_rem_mapping = pd.read_csv(os.path.join(reference_rem_path,'rem_mapping.csv'))
            self.reference_rem_mapping = self.reference_rem_mapping.set_index('transmitter_loc')


    def __len__(self):
        return self.n_terrains

    def __getitem__(self, item):

        item = item+1
        row = self.mapping.loc[item]

        terrain = get_terrain_from_info(self.terrain_info[int(row['terrain'])]['building_info'])
        rem = np.load(os.path.join(self.rem_path,f'{item}.npy'))

        if self.reference_path:
            reference_rem = np.load(os.path.join(
                self.reference_rem_path,
                f'{int(self.reference_rem_mapping.loc[row["transmitter_loc"]]["Id"])}.npy'
            ))

            transformed = self.transforms(image=terrain, rem=rem, reference_rem=reference_rem)
            return transformed['image'][0],transformed['rem'][0],transformed['reference_rem'][0]

        else:
            transformed = self.transforms(image=terrain, rem=rem)
            return transformed['image'][0],transformed['rem'][0]
