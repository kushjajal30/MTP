import numpy as np
import pylayers.antprop.loss as loss
import polarTransform as pt
from REMGeneration.utils import get_terrain_from_info,getBuildingSet
from multiprocessing import Pool

class REMGenerator:

    def __init__(self, Ht=None, Hr=1.5, fGHz=0.474166, K=1.3333, polar_radius=200, polar_radius_points=200, polar_angle=720, polar_order=3, ncpus = 1, signal_strength=50):

        if Ht is None:
            Ht = [60]

        self.Ht = Ht
        self.Hr = Hr
        self.fGHz = fGHz
        self.K = K

        self.polar_radius = polar_radius
        self.polar_radius_points = polar_radius_points
        self.polar_angle = polar_angle
        self.polar_order = polar_order

        phi = np.linspace(0, 2 * np.pi, self.polar_angle)[:, None]
        r = np.linspace(0, self.polar_radius, self.polar_radius_points)[None, :]
        self.x = r * np.cos(phi)
        self.y = r * np.sin(phi)

        self.ncpus = ncpus
        self.signal_strength = signal_strength

    def convertToPolar(self, terrain, center=(0, 0)):

        polar_terrain = pt.convertToPolarImage(terrain,
                                              radiusSize=self.polar_radius_points,
                                              angleSize=self.polar_angle,
                                              center=center,
                                              hasColor=False,
                                              order=self.polar_order)
        self.settings = polar_terrain[1]
        return polar_terrain

    def convertToCartesian(self, rem, settings=None):

        if not settings:
            settings = self.settings
        return pt.convertToCartesianImage(rem, settings=settings)

    def getREM(self, terrain, center,ht):

        polar_terrain, settings = self.convertToPolar(terrain, center=center)
        rem = loss.cover(X=self.x,
                          Y=self.y,
                          Z=polar_terrain,
                          Ha=ht,
                          Hb=self.Hr,
                          fGHz=self.fGHz,
                          K=self.K
                          )

        return self.convertToCartesian(self.signal_strength-rem[:, :, 0], settings=settings)[0]

    def getREMS(self, terrain_info):

        terrain = get_terrain_from_info(terrain_info)
        buildingset = getBuildingSet(terrain_info)

        center_start = len(terrain)//4
        center_end = 3*len(terrain)//4

        params = []
        for cx in range(center_start,center_end):
            for cy in range(center_start,center_end):
                if (cx,cy) not in buildingset:
                    for ht in self.Ht:
                        params.append((terrain, (cx, cy),ht))

        rems = []
        with Pool(processes=self.ncpus) as pool:

            for i in range(0,len(params),self.ncpus):
                rems += pool.starmap(self.getREM,params[i:i+self.ncpus])

        return np.array(rems)


if __name__ == '__main__':
    rem_gen = REMGenerator()
    terrain = np.zeros((100,100))
    terrain[20:40,70:90] = 40
    terrain[10:15,30:40] = 20

    remsample = rem_gen.getREM(terrain,center=(50,50),ht=50)
    print(remsample)

