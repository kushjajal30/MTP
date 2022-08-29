import json
import REMGeneration.config as config
from REMGeneration.REMGenerator import REMGenerator
from REMGeneration.TerrainGenerator import Terrain
import numpy as np
from tqdm import tqdm
import os

def generateFull(completed_terrain_json,rem_output_path='REMS'):

    terrain_generator = Terrain(config.__terrain_size__)
    rem_generator = REMGenerator(
        Ht=config.__Ht__,
        Hr=config.__Hr__,
        fGHz=config.__fGHz__,
        K=config.__K__,
        polar_radius=config.__polar_radius__,
        polar_radius_points=config.__polar_radius_points__,
        polar_angle=config.__polar_angle__,
        polar_order=config.__polar_order__,
        ncpus=config.__NCPUS__
    )

    terrain_info = terrain_generator.getTerrain(
        config.__number_of_buildings__,
        config.__building_min_width__,
        config.__building_min_length__,
        config.__terrain_size__,
        config.__min_height__,
        config.__max_height__,
        config.__building_max_width__,
        config.__building_max_length__
    )
    rems = rem_generator.getREMS(terrain_info)
    np.save(f'{rem_output_path}{os.sep}rems{os.sep}{len(completed_terrain_json)}.npy', rems)

    completed_terrain_json.append(terrain_info)
    with open(f'{rem_output_path}{os.sep}terrain_info.json','w') as f:
        json.dump(completed_terrain_json,f)


def main():

    if not os.path.exists(config.__output_path):
        os.mkdir(config.__output_path)
        os.mkdir(config.__output_path+os.sep+'rems')
        completed_terrain_json = []
        print("Starting new session.")
    elif not os.path.exists(config.__output_path+os.sep+'terrain_info.json'):
        completed_terrain_json = []
        print("Starting new session.")
    else:
        with open(config.__output_path+os.sep+'terrain_info.json') as f:
            completed_terrain_json = json.load(f)
        seen = len(completed_terrain_json)
        print(f"Found {seen} REMs adding after them")

    for _ in tqdm(range(config.__NREM__)):
        generateFull(completed_terrain_json,config.__output_path)

if __name__=='__main__':

    main()