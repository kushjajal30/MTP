import numpy as np


class PathSampler:

    def __init__(self, points, max_ppd=10, min_ppd=1):
        self.points = points
        self.max_ppd = max_ppd
        self.min_ppd = min_ppd
        self.directions = [(0,1),(0,-1),(1,0),(-1,0)]
        #np.random.seed(1101)

    def getRoads(self, terrain):

        roads = set()
        for i in range(len(terrain)):
            for j in range(len(terrain[0])):

                if terrain[i][j] == 0:
                    roads.add((i, j))

        return roads

    def getrandomdirection(self,directions):
        return np.random.choice(directions)

    def sample(self, terrain):

        road_sets = self.getRoads(terrain)
        start = list(road_sets)[np.random.randint(0,len(road_sets))]
        path = {start}
        stack = [(start, [0,1,2,3])]
        last3 = {}

        while len(path) < self.points and len(stack) > 0:

            point, directions = stack[-1]

            if len(directions) == 1:
                stack.pop()
                continue

            current_direction = self.directions[self.getrandomdirection(directions)]
            stack[-1][1].remove(self.directions.index(current_direction))

            ppd = np.random.randint(self.min_ppd, self.max_ppd)

            p = 0
            n = stack[-1][0]
            n = list(n)

            while True:

                n[0], n[1] = n[0] + current_direction[0], n[1] + current_direction[1]

                if n[0] < 0 or n[1] < 0 or n[0] >= len(terrain) or n[1] >= len(terrain):
                    break
                if tuple(n) in path or tuple(n) not in road_sets:
                    break

                p += 1
                path.add(tuple(n))
                if p >= ppd:
                    break

            n[0], n[1] = n[0] - current_direction[0], n[1] - current_direction[1]
            if (n[0] != stack[-1][0][0] or n[1] != stack[-1][0][1]) and n[0]!=0 and n[1]!=0 and n[0]<len(terrain)-1 and n[1]<len(terrain[0])-1:

                stack.append((tuple(n), [0,1,2,3]))
            #print(stack)
        print(len(path))
        return path

if __name__ == '__main__':
    ter = np.zeros((100,100))
    ter[:,50:]=1
    sampler = PathSampler(150)
    print(sampler.sample(ter))
