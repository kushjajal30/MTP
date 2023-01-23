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

            if len(directions) == 0:
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


class PathSampler2:

    def __init__(self, points ):
        self.points = points

    def getRoads(self, terrain):

        roads = set()
        for i in range(len(terrain)):
            for j in range(len(terrain[0])):

                if terrain[i][j] == 0:
                    roads.add((i, j))

        return roads

    def getpath(self, end, path, seen, road_sets, terrain):
        current = path[-1]
        seen.add(path[-2])

        if current not in road_sets or current[0] < 0 or current[1] < 0 or current[0] >= len(terrain) or current[
            1] >= len(terrain[0]) or current in seen:
            return False, path
        if current == end or len(seen) >= len(terrain):
            return True, path

        nexts = [
            (current[0] + 1, current[1]),
            (current[0] - 1, current[1]),
            (current[0], current[1] + 1),
            (current[0], current[1] - 1),
        ]

        for n in sorted(nexts, key=lambda a: abs(a[0] - end[0]) + abs(a[1] - end[1])):
            out, new_path = self.getpath(end, path + [n], seen, road_sets, terrain)
            if out:
                return True, new_path

        return False, path

    def modifyterrain(self,terrain,terrain_info):

        pad = np.random.choice([i for i in range(6)],p=[1/9,1/6,2/9,2/9,1/6,1/9])

        for building in terrain_info:

            i,j = max(0,building['x']-pad),max(0,building['y']-pad)

            for k in range(i,min(len(terrain),building['x']+building['length']+pad)):
                for l in range(j,min(len(terrain[0]),building['y']+building['width']+pad)):
                    terrain[k,l] += 1
        return terrain



    def sample(self, terrain,terrain_info):

        terrain = self.modifyterrain(np.copy(terrain),terrain_info)
        road_sets = self.getRoads(terrain)

        start = list(road_sets)[np.random.randint(0, len(road_sets))]
        path = [start]

        while True:

            end = list(road_sets)[np.random.randint(0, len(road_sets))]

            gotans, new_path = self.getpath(end, [None, start], set(), road_sets, terrain)
            if gotans:
                path += new_path[2:]
                start = path[-1]

            if len(path) >= self.points:
                break

        print(len(path), len(set(path)))
        return path[1:],terrain

if __name__ == '__main__':
    ter = np.zeros((100,100))
    ter[:,50:]=1
    sampler = PathSampler(150)
    print(sampler.sample(ter))
