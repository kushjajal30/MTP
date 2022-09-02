import numpy as np
from REMGeneration import config

def get_terrain_from_info(terrain_info):

    terrain = np.zeros((config.__terrain_size__,config.__terrain_size__))

    for info in terrain_info:
        terrain[info['x']:info['x']+info['length'],info['y']:info['y']+info['width']]=info['height']

    return terrain


def getBuildingSet(terrain_info):
    buildings = set()
    for info in terrain_info:
        for i in range(info['x'],info['x']+info['length']):
            for j in range(info['y'],info['y'] + info['width']):
                buildings.add((i,j))
    return buildings
