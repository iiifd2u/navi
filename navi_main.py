import matplotlib.pyplot as plt
import matplotlib.collections as mc
import time
import numpy as np
import random
import math
from optparse import OptionParser
import inspect
from typing import List
import tqdm
import pandas as pd

#METHODS:
drawing =True
learning =False
N =40#Number of points

class Point():
    def __init__(self, idx,  x=0, y=0):
        self.__index = idx
        self.__x = x
        self.__y = y
    @classmethod
    def check_types(cls, a):
        return type(a) in (float, int)

    def set_coords(self, x, y):
        if self.check_types(x) and self.check_types(y):
            self.__x = x
            self.__y = y
        else:
            raise TypeError('Некорретный тип данных')
    def get_coords(self):
        return self.__x, self.__y, self.__index

class Weight():
    def __init__(self, r):
        self.r = r

# Наличие/отсутсвие связи между точками
def link(param)->bool:
    return random.normalvariate(0, 1) > param

#Получить координаты средней тчк в линии (для подписи)
def get_middlepoint(pair:List[List])->List:
    return [(pair[0][0]+pair[1][0])/2, (pair[0][1]+pair[1][1])/2]

#Получение цвета из массива цветов по индексу
def get_color_by_number(num, colors:List, min=0, max=10):
    board = ((max-min)/max)*num/len(colors)
    return colors[math.floor(board)]

#Получение списка ширин для всей сети
def line_width_path(path:List[int], city)->List:
    normal_path = [0.5 for _ in range(len(city))]
    while len(path)>=2:
        step = path[-2:]
        path = path[:-1]
        for idx in range(len(city)):
            if city[idx][-1]==step or city[idx][-1]==step[::-1]:
                normal_path[idx]=2
    return normal_path

#Алгоритм Дейкстры - поиск соседних вершин:
def neighbors(v, D):
    for i, weight in enumerate(D[v]):
        if weight>0:
            yield i

#Алгоритм Дейкстры - выбор следующей вершины:
def argmin(T, S):
    amin = -1
    m= max(T)
    for i, t in enumerate(T):
        if i not in S and t<m:
            amin=i
            m=t
    return amin

#Алгоритм Дейкстры:
def deikstra_algorithm(D):
    N = len(D)
    last_row = [math.inf]*N
    versh =0
    set_versh = {versh}
    last_row[versh] = 0
    # список вершин вершины:
    info_list=[[] for _ in range(N)]

    while versh !=-1:
        for el in neighbors(versh, D):
            if el not in set_versh:
                w = last_row[versh]+D[versh][el]
                if last_row[el]>w:
                    last_row[el] = w
                    # добавляем к элементу вершину, из которой он пришел:
                    info_list[el].append(versh)
        versh = argmin(last_row, set_versh)
        if versh>0:
            set_versh.add(versh)
    # Словарь путей от 0 до каждой вершины:
    di = dict()
    # инициализируем словарь, каждое значение заполняется с конца:
    for idx, el in enumerate(info_list):
        di[idx] = el
    #Нужный нам элемент:
    end =8
    list_path = [end]
    # Предыдущий элемент:
    prev = end
    while prev != 0:
        # Предыдущий элемент:
        prev = di[prev][-1]
        list_path.append(prev)
    list_path.reverse()
    return (list_path, last_row[end])



points = []
#Рандом расположения городов
for i in range(N):
    points.append(Point(i, random.randint(1, 100), random.randint(1, 100)))

#Рандом наличия связи между городами
matrix_city = np.zeros((N, N))
for i in range(N):
   for j in range(i, N):
       if link(0.6) and i!=j:
           matrix_city[i][j] = matrix_city[j][i] = random.randint(1, 10)

#Рандом весов связи между городами
def fill_matrix_weights(mat_city:np.ndarray)->np.ndarray:
    matrix_weights = np.zeros((N, N))
    for i in range(N):
       for j in range(i, N):
           if mat_city[i][j]>0 and i!=j:
               matrix_weights[i][j] = matrix_weights[j][i] = random.randint(1, 8)
    return matrix_weights


#Сдвиг двумерного массива в одномерный
def one_dim_from_two(matrix:np.ndarray, city)->List:
    output=[[] for _ in city]
    for idx, el in enumerate(city):
        if matrix[el[-1][0]][el[-1][1]] >0:
            output[idx]=matrix[el[-1][0]][el[-1][1]]
        elif matrix[el[-1][1]][el[-1][1]] >0:
            output[idx]=matrix[el[-1][1]][el[-1][0]]

    return output


#Координаты для отображения
# xs = [points[i].get_coords()[0] for i in range(N)]
# ys = [points[i].get_coords()[1] for i in range(N)]

#Координаты для круга:
N1=N//2
N2 =N-N1

yc1=xc1  =2.5*N1
R1=2.5*N1
step1 =10
rang1 = 10

yc2=xc2  =2.5*N2
R2=N1
step2 =4*R2//N2
rang2 = 5

xs1 = [el for el in range(0,N1//2*step1,step1)]*2
xs2 = [el for el in range(int(xc2-R2+1),int(xc2+R2-1),step2)]*2
ys1 = [yc1-math.sqrt(R1**2 -(x-xc1)**2)+np.random.uniform(-rang1, rang1)
      for x in xs1[:len(xs1)//2]]+[yc1+math.sqrt(R1**2 -(x-xc1)**2)+np.random.uniform(-rang1, rang1)
                                 for x in xs1[len(xs1)//2:]]
ys2 = [yc2-math.sqrt(R2**2 -(x-xc2)**2)+np.random.uniform(-rang2, rang2)
      for x in xs2[:len(xs2)//2]]+[yc2+math.sqrt(R2**2 -(x-xc2)**2)+np.random.uniform(-rang2, rang2)
                                 for x in xs2[len(xs2)//2:]]
xs = xs1+xs2
# print(len(xs))
ys = ys1+ys2

# Список линий между городами
from navi_delaunay import delaunay_func, triangles_to_lines

# Делаем разбиение Делоне для группы точек на карте:
triangles = delaunay_func(list(zip(xs, ys)))

# Линии, соедиеняющие две точки на карте
lines_cities = triangles_to_lines(triangles)

# Возвращает индекс точки по её координате:
def return_idx_by_coords(pts, city:List[List]):
    for el in city:
        el.append([])
        for i in range(N):
            # print('i=', i)
            if pts[i][0]==el[0][0] and pts[i][1]==el[0][1]:
                el[2].append(i)
            if pts[i][0]==el[1][0] and pts[i][1]==el[1][1]:
                el[2].append(i)
    # Возвращаем только те значения, которые соединяют обе точки на карте
    city = list(filter(lambda x:len(x[2])>1, city))

    return city
# Коллекция неуникальных линий:
lines_not_unique=return_idx_by_coords(list(zip(xs, ys)), lines_cities)

# Переопределение в уникальный список линий
lines_cities = []
for el in lines_not_unique:
    if el not in lines_cities and [el[1], el[0], el[2]] not in lines_cities:
        lines_cities.append(el)



if drawing:
    plt.ion()
    fig, axis = plt.subplots()
    fig.set_facecolor((0.1, 0.1, 0.1))
    axis.set_facecolor((0.2, 0.2, 0.2))
    #Список цветов для отображения
    colors =['green', 'yellow', 'red']
    #Отрисовка списка узлов и связей между ними, подпись узлов
    cities = [[x, y] for x, y in zip(xs, ys)]
    axis.scatter(xs, ys, c='r')
    lc = mc.LineCollection([el[:-1] for el in lines_cities], colors=['green'], linewidths=[0.5])
    axis.add_collection(lc)

    for i in range(len(cities)):
        axis.text(cities[i][0], cities[i][1], str(i), color = 'yellow')

    middle_points = [get_middlepoint(x[:-1])  for x in lines_cities]
    mp_x = [el[0] for el in middle_points]
    mp_y = [el[1] for el in middle_points]

    text_labels =[]
    for i in range(len(middle_points)):
        text_labels.append(axis.text(mp_x[i], mp_y[i], '1', fontsize='10', color = 'white'))



T=400

weights = np.array([])
inputs =[]
outputs_no_formated = []

with tqdm.trange(T, desc='рассчёт точек') as t:
    for iteration in t:
        #Рандомные расстояния между точками на карте:
        matrix_weights = np.zeros((N, N))
        for el in lines_cities:
            i = el[-1][1]
            j = el[-1][0]
            matrix_weights[i][j] = matrix_weights[j][i] = random.randint(1, 8)
        random_weights = one_dim_from_two(matrix_weights, lines_cities)

        # Данные для обучения сети: входные данные из матрицы расстояний:
        ipts = []
        for i in range(N):
            for j in range(i, N):
                ipts.append(matrix_weights[i][j])
        inputs.append(ipts)
        # выходные данные в неформатированном виде:
        deikstra = deikstra_algorithm(matrix_weights)
        short_path =deikstra[0]
        outputs_no_formated.append(short_path)

        if drawing:
            # Шляпа для отрисовки:

            plt.ion()
            axis.set_title(short_path, color='white')
            axis.set_label('ahaha')
            colored_lines = [get_color_by_number(random_weights[x], colors)
                             for x in range(len(middle_points))]
            lc.set_linewidths(line_width_path(short_path, lines_cities))

            lc.set_color(colored_lines)
            for j in range(len(middle_points)):
                text_labels[j].set_text(str(int(random_weights[j])))

            plt.draw()
            plt.gcf().canvas.flush_events()
            time.sleep(5)


def turn_outputs_readable(inputs:List[List])->List[List]:
    outputs=[]
    for el in inputs:
        outputs.append(list(map(lambda x:1 if x in el else 0,list(range(N)))))
    return outputs

def readable_decoder(line:List[float]):
    outputs =[]
    for idx, el in enumerate(line):
        if el>0.6:
            outputs.append(idx)
    return outputs

percentile =0.75

outputs_no_formated_test=outputs_no_formated[int(percentile*T):]
outputs = turn_outputs_readable(outputs_no_formated)

outputs_train = outputs[:int(percentile*T)]
inputs_train = inputs[:int(percentile*T)]
inputs_test = inputs[int(percentile*T):]
outputs_test = outputs[int(percentile*T):]


if learning:
    from navi_neural import back_propagation, test_network
    # Обучение сетки:
    network = back_propagation(np.array(inputs_train), np.array(outputs_train), iterations=1500)
    #Тестирование сетки
    predicts = test_network(network, np.array(inputs_test),np.array(outputs_test), outputs_no_formated_test)



