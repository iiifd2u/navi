import heapq
import os
import sys
import time
import random
from termcolor import colored
from typing import Literal
from utils.metrics import manhattan_distance
from utils.config import SEP
import functools

# print = functools.partial(print, flush=True)

Color = Literal[
    'red',
    'green',
    'yellow',
    'blue'
]

def get_color(num)->Color:
    if num<=3:
        return 'green'
    elif num<=7:
        return 'yellow'
    return 'red'


N = 30

def add_streets(town, aval):
    vertical = random.sample(random.choice([list(range(0, N, 2)), list(range(1, N, 2))]),
                             random.randint(N // 3, N // 2))
    horizontal = random.sample(random.choice([list(range(0, N, 2)), list(range(1, N, 2))]),
                               random.randint(N // 3, N // 2))

    for h in horizontal:
        for i in range(N):
            town[h][i] = '.'
            aval.add(h * N + i)
    for i in range(N):
        for v in vertical:
            town[i][v] = '.'
            aval.add(i * N + v)

def dropout_buildings(town, aval):
    buildings = set(range(N * N)) - aval
    for pos in buildings:
        h, v = pos // N, pos % N
        coin = random.random()
        if coin > 0.8:
            town[h][v] = '.'
            aval.add(pos)

def add_forest(town, aval, pos:tuple, prob=1.):
    if prob<=0: return
    r, c = pos

    for i in [r-1, r, r+1]:
        if i < 0 or i > N - 1: continue
        for j in [c-1, c, c+1]:
            if j<0 or j>N-1: continue
            coin = random.random()
            if coin<prob:
                if town[i][j] == 'F': continue
                town[i][j] = 'F'
                aval.discard(i*N+j)
                add_forest(town, aval, (i, j), prob-0.25)

def add_bridges(town, aval, river, bridges = 3):
    while bridges>0:
        idx = random.choice(range(len(river)))
        pos = river.pop(idx)
        if not river:break
        i, j = pos
        if i>1 and i<N-1 and j>1 and j<N-1:
            if (town[i-1][j] == '.' and town[i+1][j] == '.') or \
                    (town[i][j-1] == '.' and town[i][j+1] == '.'):
                town[i][j] = '.'
                aval.add(i * N + j)
                bridges-=1


def add_river(town, aval:set):

    start = random.choice(list(set([(0,i) for i in range(N//2)]).union([(i,0) for i in range(N//2)])))
    r, c = start
    river = []

    while r<N and c<N:

        town[r][c] = 'R'
        aval.discard(r * N + c)
        river.append((r,c))

        coin = random.random()
        if coin>=0.67:
            r+=1
        elif coin >= 0.33:
            r+=1
            c+=1
        else:
            c+=1

    for pos in river:
        r, c = pos
        for i in [r - 1, r, r + 1]:
            if i < 0 or i > N - 1: continue
            for j in [c - 1, c, c + 1]:
                if j < 0 or j > N - 1: continue
                if town[i][j] != 'R':
                    town[i][j] = '.'
                    aval.add(i*N+j)
    add_bridges(town, aval, river, bridges=3)

def add_traffic(town, aval):
    for pos in aval:
        r, c = pos//N, pos%N
        neighbours = []
        for i in [r - 1, r, r + 1]:
            if i < 0 or i > N - 1: continue
            for j in [c - 1, c, c + 1]:
                if j < 0 or j > N - 1: continue
                if town[i][j] == '.':
                    if not neighbours:
                        coin = random.randint(0, 9) #Если соседей нет, трафик любой
                    else:
                        # иначе только близкий к соседнему
                        coin = random.randint(max(0, min(neighbours)-1), min(9, max(neighbours)+1))
                    town[i][j] = str(coin)
                    neighbours.append(coin)

def change_traffic(town, aval, threshold=2):
    for pos in aval:
        r, c = pos//N, pos%N
        val = int(town[r][c])
        # +-1 от текущего значения
        new_val = random.randint(max(0, val-threshold), min(9, val+threshold))
        town[r][c] = str(new_val)
    return town, aval





def create_town(N):

    town = [['B' for _ in range(N)] for _ in range(N)]
    aval = set()

    # add streets:
    add_streets(town, aval)

    # dropout houses
    dropout_buildings(town, aval)

    #add forest
    for _ in range(5):
        f_point = (random.randint(0, N), random.randint(0, N))
        add_forest(town, aval, f_point)

    #add river
    add_river(town, aval)

    # # add traffic
    add_traffic(town, aval)
    return town, aval


def a_star(town, start, end):
    '''классический алгоритм, учитывает текущее значение в ячейке дороги'''

    cost = {start: 0}
    came_from = {start: None}
    digits = [str(el) for el in range(10)]

    priority_queue = [(0, start)]

    while priority_queue:

        current_cost, current_point = heapq.heappop(priority_queue)
        if current_point == end:
            break

        for neighbor in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            x, y = current_point[0] + neighbor[0], current_point[1] + neighbor[1]

            new_cost = cost[current_point] + int(town[current_point[0]][current_point[1]])

            if 0 <= x < len(town) and 0 <= y < len(town[0]) and town[x][y] in digits:
                if (x, y) not in cost or new_cost < cost[(x, y)]:
                    cost[(x, y)] = new_cost
                    priority = new_cost + manhattan_distance((x, y), end) # Приоритетное направление

                    heapq.heappush(priority_queue, (priority, (x, y)))
                    came_from[(x, y)] = current_point
    path = []
    current = end
    while current:
        path.append(current)
        current = came_from[current]
    path.reverse()

    return path

def pretty_print(town, path):
    N = len(town)
    for i in range(N):
        for j in range(N):
            char = town[i][j]
            if char=='F':
                print(colored(char, 'green', on_color='on_light_green', attrs=['bold']), end=SEP)
            elif char=='R':
                print(colored(char, 'blue', on_color='on_cyan', attrs=['bold']), end=SEP)
            elif char == 'B':
                print(colored(char, 'black',on_color='on_light_grey', attrs=['bold']), end=SEP)
            else:
                num = int(char)
                if (i, j) in path:
                    if (i, j) == path[0]:
                        char = 'S'
                    elif (i, j) == path[-1]:
                        char = 'E'
                    else:
                        char = '#'
                    print(colored(char, color=get_color(num),attrs=['bold']), end=SEP)
                else:
                    print(colored('.',  get_color(num)), end=SEP)
        print()



if __name__ == '__main__':
    # random.seed(1)
    town, aval = create_town(N)

    start_idx = random.choice(list(aval))
    end_idx = random.choice(list(aval))
    start = (start_idx//N, start_idx%N)
    end = (end_idx//N, end_idx%N)

    path = a_star(town, start, end)
    while True:
        pretty_print(town, path)
        town, aval = change_traffic(town, aval)
        path = a_star(town, start, end)
        time.sleep(1)
        os.system('cls')