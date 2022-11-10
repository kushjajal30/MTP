import numpy as np
import os
import time

class Terrain:
    def __init__(self, terrain_size):
        np.random.seed((os.getpid() * int(time.time())) % 123456789)
        self.terrain_size = terrain_size

    def getTerrain(self, number_of_buildings, building_min_width, building_min_length, terrain_size, min_height, max_height,  building_max_width, building_max_length):
        terrain_info = []
        number_of_buildings_each_axis = np.sqrt(number_of_buildings)
        gap = (terrain_size / number_of_buildings)/2 #gap = (terrain_size / number_of_buildings_each_axis) / number_of_buildings_each_axis
        number_of_buildings_y_axis = np.random.randint(2, number_of_buildings_each_axis)
        number_of_buildings_x_axis = np.random.randint(2, number_of_buildings_each_axis)
        max_width_of_building_x_axis = min((terrain_size / number_of_buildings_x_axis), building_max_width)
        max_length_of_building_y_axis = min((terrain_size / number_of_buildings_x_axis), building_max_length)
        #max_length_of_building_y_axis =  terrain_size / number_of_buildings_y_axis
        x_unit = int(terrain_size / number_of_buildings_x_axis)
        y_unit = int(terrain_size / number_of_buildings_y_axis)

        start = [0,0]
        for i in range(number_of_buildings_x_axis):
            start[0] = i * x_unit #+ gap
            start[1] = 0
            for j in range(number_of_buildings_y_axis):
                start[1] = j * y_unit# + gap
                building_width = np.random.randint(building_min_width, max_width_of_building_x_axis)
                building_length = np.random.randint(building_min_length, max_length_of_building_y_axis)

                start_x_max = (start[0] + x_unit) - building_width
                #print("start[0] : ",start[0],"\t\tstart_x_max : ",start_x_max)

                start_x = np.random.randint(start[0], start_x_max)
                if(building_length >= y_unit):
                    building_length = y_unit - 3
                start_y_max = (start[1] + y_unit) - building_length
                #print("start[1] : ",start[1],"\t\tstart_y_max : ",start_y_max)
                start_y = np.random.randint(start[1], start_y_max)
                #print("start_x : ",start_x,"\t\tstart_y : ",start_y)
                height = np.random.randint(min_height, max_height)
                #For this module width is connected to x while other modules length is connected to x
                terrain_info.append(
                    {'x': start_x, 'y': start_y, 'length': building_width, 'width': building_length, 'height': height}
                )
                #self.get_building(start_x, start_y, building_width, building_length, height)

        return terrain_info

    def getTerrain2(self, average_number_of_building):

        terrain_info = []

        number_of_buildings = round(np.random.normal(loc=average_number_of_building,scale=average_number_of_building/4))

        for i in range(number_of_buildings):

            building_width = min(max(round(np.random.normal(loc=30,scale=10)),10),60)
            building_length = min(max(round(np.random.normal(loc=30,scale=10)),10),60)
            building_height = np.random.randint(10,40)

            x = np.random.randint(5,self.terrain_size-30)
            y = np.random.randint(5,self.terrain_size-30)

            terrain_info.append(
                {'x': x, 'y': y, 'length': building_width, 'width': building_length, 'height': building_height}
            )


        return terrain_info
if __name__ == '__main__':
    number_of_buildings = 20
    building_min_width = 10
    building_min_length = 10
    building_max_width = 60
    building_max_length = 100
    terrain_size = 300
    min_height = 15
    max_height = 40

    terr = Terrain(terrain_size)

    terrain = terr.getTerrain(number_of_buildings, building_min_width, building_min_length, terrain_size, min_height, max_height,
                    building_max_width, building_max_length)
    print(terrain)