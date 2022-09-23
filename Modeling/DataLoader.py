from torch.utils.data import Dataset
from REMGeneration.utils import get_terrain_from_info,getBuildingSet
from REMGeneration.REMGenerator import REMGenerator
from REMGeneration.TerrainGenerator import Terrain
import os
import pandas as pd
import numpy as np
import json
import random

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
            return transformed['image'],transformed['rem'],transformed['reference_rem']

        else:
            transformed = self.transforms(image=terrain, rem=rem)
            return transformed['image'],transformed['rem']


class REMInTimeDataset(Dataset):

    def __init__(self,config,terrain_per_epoch,crop_center,transforms):
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
        self.crop_center = crop_center

        self.transforms = transforms

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

        buildingset = getBuildingSet(terrain_info)
        roadset = []
        center_start = len(terrain)//4
        center_end = 3*len(terrain)//4

        for i in range(self.config.__terrain_size__):
            for j in range(self.config.__terrain_size__):
                if center_start <= i < center_end and center_start <= j < center_end and (i, j) not in buildingset:
                    roadset.append((j,i))

        center = random.choice(roadset)
        ht = random.choice(self.config.__Ht__)

        if self.crop_center:
            terrain = terrain[center[1]-len(terrain)//4:center[1]+len(terrain)//4,center[0]-len(terrain)//4:center[0]+len(terrain)//4]

        rem = self.rem_generator.getREM(terrain,(len(terrain)//2,len(terrain)//2),ht)


        transformed = self.transforms(image=terrain, rem=rem)
        return transformed['image'], transformed['rem']


