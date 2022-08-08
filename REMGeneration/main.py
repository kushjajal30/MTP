import REMGeneration.config as config
from REMGeneration.REMGenerator import REMGenerator
from REMGeneration.TerrainGenerator import Terrain
from multiprocess.pool import Pool
from itertools import repeat
from tqdm import tqdm
import os

def generateFull(terrain_generator,rem_generator,i):

    terrain = terrain_generator.getTerrain(
        config.__number_of_buildings__,
        config.__building_min_width__,
        config.__building_min_length__,
        config.__terrain_size__,
        config.__min_height__,
        config.__max_height__,
        config.__building_max_width__,
        config.__building_max_length__
    )

    rem_generator.getREMS(terrain,i=i,save=True)

def main():

    n_cpus = os.cpu_count()
    batches = config.__NREM__//n_cpus
    print(f"Found {n_cpus} cpus, Starting Generation of {config.__NREM__} Data Points.")

    if not os.path.exists(config.__output_path):
        os.mkdir(config.__output_path)
        os.mkdir(config.__output_path+os.sep+'rems')
        os.mkdir(config.__output_path+os.sep+'terrains')
        seen = 0
        print("Created New Directory.")
    else:
        seen = len(os.listdir(config.__output_path+os.sep+'rems'))
        print(f"Found {seen} REMs adding after them")

    terrain_generator = Terrain(config.__terrain_size__)
    rem_generator = REMGenerator(
        Ht=config.__Ht__,
        Hr=config.__Hr__,
        fGHz=config.__fGHz__,
        K=config.__K__,
        polar_radius=config.__polar_radius__,
        polar_radius_points=config.__polar_radius_points__,
        polar_angle=config.__polar_angle__,
        polar_order=config.__polar_order__
    )

    if batches==0:
        batches=1

    with Pool() as pool:

        for batch in tqdm(range(batches)):
            pool.starmap(generateFull,zip(
                repeat(terrain_generator),
                repeat(rem_generator),
                range(seen+batch*n_cpus,seen+(batch+1)*n_cpus)
            ))

if __name__=='__main__':

    main()