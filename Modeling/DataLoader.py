from torch.utils.data import Dataset
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from REMGeneration.utils import get_terrain_from_info
from REMGeneration.REMGenerator import REMGenerator
from REMGeneration.TerrainGenerator import Terrain
import os
import pandas as pd
import numpy as np
import json
import random


class REMStaticDataset(Dataset):

    def __init__(self, path, transforms, reference_rem_path=None):

        self.path = path
        self.rem_path = os.path.join(path, 'rems')

        with open(os.path.join(path, 'terrain_info.json'), 'r') as f:
            self.terrain_info = json.load(f)

        self.mapping = pd.read_csv(os.path.join(path, 'rem_mapping.csv'))
        self.mapping = self.mapping.set_index('Id')

        self.n_terrains = len(os.listdir(self.rem_path))
        self.transforms = transforms
        self.reference_path = reference_rem_path

        if reference_rem_path:
            self.reference_path = reference_rem_path
            self.reference_rem_path = os.path.join(reference_rem_path, 'rems')
            self.reference_rem_mapping = pd.read_csv(os.path.join(reference_rem_path, 'rem_mapping.csv'))
            self.reference_rem_mapping = self.reference_rem_mapping.set_index('transmitter_loc')

    def __len__(self):
        return self.n_terrains

    def __getitem__(self, item):

        item = item + 1
        row = self.mapping.loc[item]

        terrain = get_terrain_from_info(self.terrain_info[int(row['terrain'])]['building_info'])
        rem = np.load(os.path.join(self.rem_path, f'{item}.npy'))

        if self.reference_path:
            reference_rem = np.load(os.path.join(
                self.reference_rem_path,
                f'{int(self.reference_rem_mapping.loc[row["transmitter_loc"]]["Id"])}.npy'
            ))

            transformed = self.transforms(image=terrain, rem=rem, reference_rem=reference_rem)
            return transformed['image'], transformed['rem'], transformed['reference_rem']

        else:
            transformed = self.transforms(image=terrain, rem=rem)
            return transformed['image'], transformed['rem']


class REMInTimeDataset(Dataset):

    def __init__(self, config, terrain_per_epoch,eigen_rems=None,rem_high=40, rem_low=-60):
        self.config = config
        self.terrain_generator = Terrain(config.__terrain_size__)
        self.rem_generator = REMGenerator(
            Ht=config.__Ht__,
            Hr=config.__Hr__,
            fGHz=config.__fGHz__,
            K=config.__K__,
            polar_radius=config.__polar_radius__,
            polar_radius_points=config.__polar_radius_points__,
            polar_angle=config.__polar_angle__,
            polar_order=config.__polar_order__,
            ncpus=config.__NCPUS__,
            signal_strength=config.__signal_strength__
        )
        self.terrain_per_epoch = terrain_per_epoch
        terrain = np.zeros((config.__terrain_size__, config.__terrain_size__))
        self.rem_value_range = (rem_low, rem_high)

        self.transforms = Compose([
            ToTensorV2(),
        ], additional_targets={'rem': 'image'}
        )
        self.eigen_rems = eigen_rems

    def __len__(self):
        return self.terrain_per_epoch

    def __getitem__(self, item):
        terrain_info = self.terrain_generator.getTerrain(
            self.config.__number_of_buildings__,
            self.config.__building_min_width__,
            self.config.__building_min_length__,
            self.config.__terrain_size__,
            self.config.__min_height__,
            self.config.__max_height__,
            self.config.__building_max_width__,
            self.config.__building_max_length__
        )

        terrain = get_terrain_from_info(terrain_info)
        ht = random.choice(self.config.__Ht__)
        rem = self.rem_generator.getREM(terrain, (0, 0), ht)

        rem = np.clip(rem, a_min=self.rem_value_range[0], a_max=self.rem_value_range[1])
        rem = (rem - self.rem_value_range[0]) / (self.rem_value_range[1] - self.rem_value_range[0])

        if self.eigen_rems:
            rem-=self.eigen_rems

        transformed = self.transforms(image=terrain, rem=rem)
        return transformed['image'] / self.config.__max_height__, transformed['rem']