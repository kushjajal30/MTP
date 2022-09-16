import json
import pandas as pd
import REMGeneration.config as config
from REMGeneration.REMGenerator import REMGenerator
from REMGeneration.TerrainGenerator import Terrain
import numpy as np
from tqdm import tqdm
import os

def generateFull(completed_terrain_json,completed_df,rem_output_path,seen):

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
        ncpus=config.__NCPUS__,
        signal_strength=config.__signal_strength__
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
    rems,params = rem_generator.getREMS(terrain_info)

    ids = []

    for rem in rems:
        seen+=1
        ids.append(seen)
        np.save(f'{rem_output_path}{os.sep}rems{os.sep}{seen}.npy', rem)

    addition_df = pd.DataFrame({'Id':ids,'terrain':[len(completed_terrain_json)]*len(ids),'transmitter_loc':params})
    completed_df = pd.concat([completed_df,addition_df],axis=0)
    completed_df.to_csv(config.__output_path+os.sep+'rem_mapping.csv',index=False)

    completed_terrain_json.append({'building_info':terrain_info})
    with open(f'{rem_output_path}{os.sep}terrain_info.json','w') as f:
        json.dump(completed_terrain_json,f)

    return completed_terrain_json,completed_df,seen


def main():

    if not os.path.exists(config.__output_path):
        os.mkdir(config.__output_path)
        os.mkdir(config.__output_path+os.sep+'rems')
        completed_terrain_json = []
        completed_df = pd.DataFrame({'Id':[],'terrain':[],'transmitter_loc':[]})
        seen = 0
        print("Starting new session.")
    elif not os.path.exists(config.__output_path+os.sep+'terrain_info.json'):
        completed_terrain_json = []
        completed_df = pd.DataFrame({'Id':[],'terrain':[],'transmitter_loc':[]})
        seen = 0
        print("Starting new session.")
    else:
        with open(config.__output_path+os.sep+'terrain_info.json') as f:
            completed_terrain_json = json.load(f)
        completed_df = pd.read_csv(config.__output_path+os.sep+'rem_mapping.csv')
        seen = len(completed_df)
        print(f"Found {seen} REMs adding after them")

    for _ in tqdm(range(config.__NDEM__)):
        completed_terrain_json,completed_df,seen = generateFull(completed_terrain_json,completed_df,config.__output_path,seen)

if __name__=='__main__':

    main()