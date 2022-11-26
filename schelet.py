import random, math
from _functools import reduce
from copy import copy, deepcopy
from builtins import isinstance
from resource import setrlimit, RLIMIT_AS, RLIMIT_DATA

import sys
from itertools import product
from heapq import heappush, heappop
from statistics import mean, variance
import time

class NPuzzle:
    """
    Reprezentarea unei stări a problemei și a istoriei mutărilor care au adus starea aici.

    Conține funcționalitate pentru
    - afișare
    - citirea unei stări dintr-o intrare pe o linie de text
    - obținerea sau ștergerea istoriei de mutări
    - obținerea variantei rezolvate a acestei probleme
    - verificarea dacă o listă de mutări fac ca această stare să devină rezolvată.
    """

    NMOVES = 4
    UP, DOWN, LEFT, RIGHT = range(NMOVES)
    ACTIONS = [UP, DOWN, LEFT, RIGHT]
    names = "DOWN, UP, RIGHT, LEFT".split(", ")
    BLANK = ' '
    delta = dict(zip(ACTIONS, [(-1, 0), (1, 0), (0, -1), (0, 1)]))

    PAD = 2

    def __init__(self, puzzle : list[int | str], movesList : list[int] = []):
        """
        Creează o stare nouă pe baza unei liste liniare de piese, care se copiază.

        Opțional, se poate copia și lista de mutări dată.
        """
        self.N = len(puzzle)
        self.side = int(math.sqrt(self.N))
        self.r = copy(puzzle)
        self.moves = copy(movesList)

    def display(self, show = True) -> str:
        l = "-" * ((NPuzzle.PAD + 1) * self.side + 1)
        aslist = self.r

        slices = [aslist[ slice * self.side : (slice+1) * self.side ]  for slice in range(self.side)]
        s = ' |\n| '.join([' '.join([str(e).rjust(NPuzzle.PAD, ' ') for e in line]) for line in slices])

        s = ' ' + l + '\n| ' + s + ' |\n ' + l
        if show: print(s)
        return s
    def display_moves(self):
        print([self.names[a] if a is not None else None for a in self.moves])

    def print_line(self):
        return str(self.r)

    @staticmethod
    def read_from_line(line : str):
        list = line.strip('\n][').split(', ')
        numeric = [NPuzzle.BLANK if e == "' '" else int(e) for e in list]
        return NPuzzle(numeric)

    def clear_moves(self):
        """Șterge istoria mutărilor pentru această stare."""
        self.moves.clear()

    def apply_move_inplace(self, move : int):
        """Aplică o mutare, modificând această stare."""
        blankpos = self.r.index(NPuzzle.BLANK)
        y, x = blankpos // self.side, blankpos % self.side
        ny, nx = y + NPuzzle.delta[move][0], x + NPuzzle.delta[move][1]
        if ny < 0 or ny >= self.side or nx < 0 or nx >= self.side: return None
        newpos = ny * self.side + nx
        piece = self.r[newpos]
        self.r[blankpos] = piece
        self.r[newpos] = NPuzzle.BLANK
        self.moves.append(move)
        return self

    def apply_move(self, move : int):
        """Construiește o nouă stare, rezultată în urma aplicării mutării date."""
        return self.clone().apply_move_inplace(move)

    def solved(self):
        """Întoarce varianta rezolvată a unei probleme de aceeași dimensiune."""
        return NPuzzle(list(range(self.N))[1:] + [NPuzzle.BLANK])

    def verify_solved(self, moves : list[int]) -> bool:
        """"Verifică dacă aplicarea mutărilor date pe starea curentă duce la soluție"""
        return reduce(lambda s, m: s.apply_move_inplace(m), moves, self.clone()) == self.solved()

    def heuristic_misplaced(self) -> int:
        return sum(map(lambda t: 1 if t[0] != t[1] else 0, zip(self.r, self.solved().r)))

    def heuristic_manhattan(self) -> int:
        dist = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
        pos = lambda n : (self.side - 1, self.side - 1) if n == ' ' else ((n - 1) // self.side, (n - 1) % self.side)
        tiles = [(i, j) for i in range(self.side) for j in range(self.side)]
        return sum(map(lambda val, tile: dist(pos(val), tile), self.r, tiles))

    def neighbours(self):
        return filter(lambda x: x, [self.apply_move(i) for i in range(self.NMOVES)])

    def astar(self, h):
        # Frontier, represented as a heap of tuples (cost + heuristic, node)
        frontier = []
        heappush(frontier, (h(self), self))
        discovered = {self.__hash__(): 0}

        # function for filtering nodes that need to be visited
        must_update = lambda x: x.__hash__() not in discovered or cost < discovered[x.__hash__()]

        best = self
        while frontier:
            _, top = heappop(frontier)
            cost = discovered[top.__hash__()] + 1

            if h(top) == 0:
                return (len(discovered), discovered[top.__hash__()], top)

            for node in filter(must_update, top.neighbours()):
                heappush(frontier, (cost + h(node), node))
                discovered[node.__hash__()] = cost

        return None

    def beamSearch(self, B, h, limit):
        beam = {self}
        visited = {self.__hash__()}
        check_visited = lambda s: s.__hash__() not in visited

        while beam and len(visited) < limit:
            succ = []
            for s in beam:
                for n in filter(check_visited, s.neighbours()):
                    heappush(succ, (h(n), n))

            if not succ:
                print('No successors left')
                return None

            score, best = heappop(succ)
            if score == 0:
                print('visited:', len(visited))
                return (len(visited), len(best.moves), best)

            selected = {best}
            for i in range(min(len(succ), B - 1)):
                _, state = heappop(succ)
                selected.add(state)

            visited.update(map(NPuzzle.__hash__, selected))
            beam = selected

        return None

    def glds_iteration(self, discrepancy, h, visited, limit):
        #print("visited:", len(visited))
        succ = []
        for s in self.neighbours():
            score = h(s)
            if score == 0:
                return (score, s)
            if s.__hash__() not in visited:
                heappush(succ, (score, s))

        if not succ or len(visited) > limit:
            return None

        _, best = heappop(succ)
        if discrepancy == 0:
            return best.glds_iteration(0, h, visited.union({best.__hash__()}), limit)
        else:
            while succ:
                _, s = heappop(succ)
                solution = s.glds_iteration(discrepancy - 1, h, visited.union({s.__hash__()}), limit)
                if solution:
                    return solution

            return best.glds_iteration(discrepancy, h, visited.union({best.__hash__()}), limit)

    def glds(self, h, limit):
        visited = {self.__hash__()}

        discrepancy = 0
        while True:
            print(discrepancy)
            solution = self.glds_iteration(discrepancy, h, visited, limit)
            if solution:
                return solution
            discrepancy += 1

        return None

    def ggg(self, h, limit):
        discrepancy = 0
        while True:
            print('DISC', discrepancy)
            stack = [(self, discrepancy, set())]
            while stack:
                state, d, visited = stack.pop()
                succ = []

                for s in state.neighbours():
                    score = h(s)
                    if score == 0:
                        return (score, s)
                    if s.__hash__() not in visited:
                        heappush(succ, (score, s))

                if not succ or len(visited) > limit:
                    continue

                _, best = heappop(succ)
                if d == 0:
                    stack.append((best, d, visited.union({best.__hash__()})))
                if d > 0:
                    tovisit = []
                    while succ:
                        _, s = heappop(succ)
                        tovisit.append((s, d - 1, visited.union({s.__hash__()})))
                    print([h(s[0]) for s in tovisit])
                    while tovisit:
                        stack.append(tovisit.pop())

            discrepancy += 1

    @staticmethod
    def blds_iteration(level, discrepancy, B, h, visited, limit):
        succ = []

        for s in level:
            for n in s.neighbours():
                score = h(n)
                if score == 0:
                    return (score, n)
                elif n.__hash__() not in visited:
                    heappush(succ, (score, n))

        if not succ or len(visited) + min(B, len(succ)) > limit:
            return None

        if discrepancy == 0:
            next_level = []
            for i in range(min(B, len(succ))):
                _, node = heappop(succ)
                next_level.append(node)

            next_visited = visited.union(map(NPuzzle.__hash__, next_level))
            return NPuzzle.blds_iteration(next_level, 0, B, h, next_visited, limit)
        else:
            explored = B
            b_level = []
            for i in range(B):
                _, node = heappop(succ)
                b_level.append(node)
            while explored < len(succ):
                n = min(len(succ) - explored, B)
                next_level = []
                for i in range(n):
                    _, node = heappop(succ)
                    next_level.append(node)
                next_visited = visited.union(map(NPuzzle.__hash__, next_level))
                val = NPuzzle.blds_iteration(next_level, discrepancy - 1, B, h, next_visited, limit)
                if val:
                    return val
                explored += len(next_level)
            b_visited = visited.union(map(NPuzzle.__hash__, b_level))
            return NPuzzle.blds_iteration(b_level, discrepancy, B, h, b_visited, limit)


    def blds(self, h, B, limit):
        visited = {self.__hash__()}
        discrepancy = 0
        while True:
            print(discrepancy)
            solution = NPuzzle.blds_iteration([self], discrepancy, B, h, visited, limit)
            if solution:
                return solution
            discrepancy += 1

    def clone(self):
        return NPuzzle(self.r, self.moves)
    def __str__(self) -> str:
        return str(self.N-1) + "-puzzle:" + str(self.r)
    def __repr__(self) -> str: return str(self)
    def __eq__(self, other):
        return self.r == other.r
    def __lt__(self, other):
        return True
    def __hash__(self):
        return hash(tuple(self.r))


class TowersOfHanoi:
    def __init__(self, towerCount, discCount, discs=[], moves=[]):
        self.towerCount = towerCount
        self.discCount = discCount
        if discs:
            self.discs = deepcopy(discs)
        else:
            self.discs = [[i for i in range(discCount - 1, -1, -1)]] + [[] for i in range(towerCount - 1)]
        if moves:
            self.moves = copy(moves)
        else:
            self.moves = []

    def move_inplace(self, src, dest):
        self.discs[dest].append(self.discs[src].pop())

        self.moves.append((src, dest))
        return self

    def move(self, src, dest):
        if src > self.towerCount or dest > self.towerCount or src < 0 or dest < 0:
            return None
        if src == dest or not self.discs[src]:
            return None
        if self.discs[dest] and self.discs[src][-1] > self.discs[dest][-1]:
            return None
        if len(self.discs[dest]) > self.discCount:
            return None

        return self.clone().move_inplace(src, dest)

    def distanceHeuristic(self):
        return sum(map(lambda d : sum([(i+1) for i in range(len(d))]), self.discs[:-1]))

    def neighbours(self):
        filledTowers = list(filter(lambda n : self.discs[n], range(self.towerCount)))
        moves = product(filledTowers, range(self.towerCount))
        return list(filter(lambda x : x, [self.move(i, j) for (i, j) in moves]))

    def display(self):
        for i in range(self.discCount):
            for j in range(self.towerCount):
                if (self.discCount - i - 1) < len(self.discs[j]):
                    print(self.discs[j][self.discCount - i - 1], end='  ')
                else:
                    print('|', end='  ')
            print('')
        print(self.towerCount * '---')

    def astar(self, h):
        # Frontier, represented as a heap of tuples (cost + heuristic, node)
        frontier = []
        heappush(frontier, (h(self), self))
        discovered = {self.__hash__(): 0}

        # function for filtering nodes that need to be visited
        must_update = lambda x: x.__hash__() not in discovered or cost < discovered[x.__hash__()]

        best = self
        while frontier:
            _, top = heappop(frontier)
            cost = discovered[top.__hash__()] + 1

            if h(top) == 0:
                return (discovered[top.__hash__()], top)

            for node in filter(must_update, top.neighbours()):
                heappush(frontier, (cost + h(node), node))
                discovered[node.__hash__()] = cost

        return None


    @staticmethod
    def blds_iteration(level, discrepancy, B, h, visited, limit):
        succ = []

        for s in level:
            for n in s.neighbours():
                score = h(n)
                if score == 0:
                    return (score, n)
                elif n.__hash__() not in visited:
                    heappush(succ, (score, n))

        if not succ or len(visited) + min(B, len(succ)) > limit:
            return None

        print("VISITED", len(visited))

        if discrepancy == 0:
            next_level = []
            for i in range(min(B, len(succ))):
                _, node = heappop(succ)
                next_level.append(node)

            next_visited = visited.union(map(TowersOfHanoi.__hash__, next_level))
            return TowersOfHanoi.blds_iteration(next_level, 0, B, h, next_visited, limit)
        else:
            explored = B
            b_level = []
            for i in range(B):
                _, node = heappop(succ)
                b_level.append(node)
            while explored < len(succ):
                n = min(len(succ) - explored, B)
                next_level = []
                for i in range(n):
                    _, node = heappop(succ)
                    next_level.append(node)
                next_visited = visited.union(map(TowersOfHanoi.__hash__, next_level))
                val = TowersOfHanoi.blds_iteration(next_level, discrepancy - 1, B, h, next_visited, limit)
                if val:
                    return val
                explored += len(next_level)
            b_visited = visited.union(map(TowersOfHanoi.__hash__, b_level))
            return TowersOfHanoi.blds_iteration(b_level, discrepancy, B, h, b_visited, limit)


    def blds(self, h, B, limit):
        visited = {self.__hash__()}
        discrepancy = 0
        for i in range(10):
            print("DISC", discrepancy)
            solution = TowersOfHanoi.blds_iteration([self], discrepancy, B, h, visited, limit)
            if solution:
                return solution
            discrepancy += 1

    def clone(self):
        return TowersOfHanoi(self.towerCount, self.discCount, self.discs, self.moves)

    def __eq__(self, other):
        return self.discs == other.discs
    def __lt__(self, other):
        return True
    def __hash__(self):
        return hash(tuple([tuple(d) for d in self.discs]))

MLIMIT = 10 * 10 ** 9 # 10 GB RAM limit
setrlimit(RLIMIT_DATA, (MLIMIT, MLIMIT))
sys.setrecursionlimit(100000)

t = 4
d = 8
hanoi = TowersOfHanoi(t, d)
#solution = hanoi.blds(TowersOfHanoi.distanceHeuristic, 100, 1000000)
solution = hanoi.astar(TowersOfHanoi.distanceHeuristic)
if solution:
    _, solved = solution
    print("SOLUTION: ", len(solved.moves), "steps")
    s = TowersOfHanoi(t, d)
    for m in solved.moves:
        s.move_inplace(*m)
        s.display()
else:
    print("NO SOLUTION FOUND")

ntests = 4
file_names = ['files/problems4-easy.txt', 'files/problems5-easy.txt', 'files/problems6-easy.txt']
print('START')
#for file_name, size in zip(file_names, [100000, 500000, 1000000]):
#    f = open(file_name, "r")
#    input = f.readlines()
#    f.close()
#    problems = [NPuzzle.read_from_line(line) for line in input]
#    algs = {'astar': (NPuzzle.astar, [NPuzzle.heuristic_manhattan]),
#            'beamB1': (NPuzzle.beamSearch, [1, NPuzzle.heuristic_manhattan, size]),
#            'beamB10': (NPuzzle.beamSearch, [10, NPuzzle.heuristic_manhattan, size]),
#            'beamB50': (NPuzzle.beamSearch, [50, NPuzzle.heuristic_manhattan, size]),
#            'beamB100': (NPuzzle.beamSearch, [100, NPuzzle.heuristic_manhattan, size]),
#            'beamB500': (NPuzzle.beamSearch, [500, NPuzzle.heuristic_manhattan, size]),
#            'beamB1000': (NPuzzle.beamSearch, [1000, NPuzzle.heuristic_manhattan, size])
#    }
#
#    results = {}
#    for (algname, v) in algs.items():
#        f, args = v
#        for i in range(len(problems)):
#            k = file_name + '_test' + str(i+1) + '_' + algname
#            print('TEST ' + k)
#            results[k] = {'cost': [], 'visited': [], 'time': []}
#            for t in range(ntests):
#                start = time.time()
#                solution = f(problems[i], *args)
#                end = time.time()
#                if solution:
#                    visited, cost, solved = solution
#                    results[k]['cost'].append(cost)
#                    results[k]['visited'].append(visited)
#                    results[k]['time'].append(end - start)
#                    print('SUCCESS', cost, visited, end - start)
#                else:
#                    results[k]['cost'].append(None)
#                    results[k]['visited'].append(None)
#                    results[k]['time'].append(None)
#                    print('FAIL')
#
#            if None in results[k]['cost']:
#                print('success rate:', len(list(filter(lambda x : x, results[k]['cost']))) / ntests)
#            else:
#                print(results[k])
#                print('mean cost:', mean(results[k]['cost']))
#                print('variance cost:', variance(results[k]['cost']))
#                print('mean visited:', mean(results[k]['visited']))
#                print('variance visited:', variance(results[k]['visited']))
#                print('mean time:', mean(results[k]['time']))
#                print('variance time:', variance(results[k]['time']))
#                print('---')
