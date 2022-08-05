import os
import numpy as np
import pylayers.antprop.loss as loss
import polarTransform as pt


class REMGenerator:

    def __init__(self, Ht=60, Hr=1.5, fGHz=0.474166, K=1.3333, polar_radius=200, polar_angle=720, polar_order=3):

        self.Ht = Ht
        self.Hr = Hr
        self.fGHz = fGHz
        self.K = K

        self.polar_radius = polar_radius
        self.polar_angle = polar_angle
        self.polar_order = polar_order

        phi = np.linspace(0, 2 * np.pi, self.polar_angle)[:, None]
        r = np.linspace(0, self.polar_radius, self.polar_radius)[None, :]
        self.x = r * np.cos(phi)
        self.y = r * np.sin(phi)

    def convertToPolar(self, terrain, center=(0, 0)):

        polar_terrain = pt.convertToPolarImage(terrain,
                                              radiusSize=self.polar_radius,
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

    def getREM(self, terrain, center):

        polar_terrain, settings = self.convertToPolar(terrain, center=center)
        rem = -loss.cover(X=self.x,
                          Y=self.y,
                          Z=polar_terrain,
                          Ha=self.Ht,
                          Hb=self.Hr,
                          fGHz=self.fGHz,
                          K=self.K
                          )

        return np.flip(self.convertToCartesian(rem[:, :, 0], settings=settings)[0], axis=0)

    def getREMS(self, terrain, rem_output_path='REMS',i=0, save=False):

        center_start = len(terrain)//4
        center_end = 3*len(terrain)//4

        rems = []
        for cx in range(center_start,center_end):
            for cy in range(center_start,center_end):
                rems.append(np.expand_dims(self.getREM(terrain, center=(cx, cy)),axis=0))
        rems = np.array(rems)

        if save:
            np.save(f'{rem_output_path}{os.sep}rems{os.sep}{i}.npy', np.concatenate(rems, axis=0))
            np.save(f'{rem_output_path}{os.sep}terrains{os.sep}{i}.npy',terrain)

        return rems


if __name__ == '__main__':
    rem_gen = REMGenerator()
    terrain = np.zeros((100,100))
    terrain[20:40,70:90] = 40
    terrain[10:15,30:40] = 20

    rem = rem_gen.getREM(terrain,center=(50,50))
    print(rem)

