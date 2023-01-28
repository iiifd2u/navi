import numpy as np
from scipy.linalg import solve
from typing import List
import math
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import pylab as pl


def get_lines(listpoints:List, itin:List)->List:
    lines=[]
    for i in range(len(itin)-1):
        lines.append([listpoints[itin[i]], listpoints[itin[i+1]]])
    return lines

def get_circumcircle(triangle):
    xA, yA = triangle[0]
    xB, yB = triangle[1]
    xC, yC = triangle[2]

    k1 = 2*(xB-xA)
    k2 = 2*(yB-yA)
    b1 = (xB**2-xA**2)+(yB**2-yA**2)

    k3 = 2 * (xC - xB)
    k4 = 2 * (yC - yB)
    b2 = (xC ** 2 - xB ** 2) + (yC ** 2 - yB ** 2)
    A=np.array([
        [k1, k2],
        [k3, k4]
    ])
    B = np.array([[b1], [b2]])
    solution =solve(A, B)
    xO, yO =  tuple(el[0] for el in solution)
    R = np.sqrt((xO-xA)**2+(yO-yA)**2)
    return ((xO, yO), R)

def triangles_by_points(pt1, pt2, pt3)->List[List]:
    return [list(pt1), list(pt2), list(pt3)]

def distance(pt1, pt2)->float:
    return math.sqrt((pt2[0]-pt1[0])**2+(pt2[1]-pt1[1])**2)


def delaunay_func(points):

    delaunay = [triangles_by_points([-500, -500], [-500, 1000], [1000, -500])]

    for pt_num in range(len(points)):
        invalid_triangles =[]
        for del_idx in range(len(delaunay)):
            center,  radius = get_circumcircle(delaunay[del_idx])
            dist = distance(center, points[pt_num])
            if dist<radius:
                invalid_triangles.append(delaunay[del_idx])

        points_in_invalid = []
        for i in range(len(invalid_triangles)):
            delaunay.remove(invalid_triangles[i])
            for j in range(3):
                points_in_invalid.append(invalid_triangles[i][j])
        points_in_invalid = [list(x) for x in set(tuple(x) for x in points_in_invalid)]

        for i in range(len(points_in_invalid)):
            for j in range(i+1, len(points_in_invalid)):
                single_poins=0
                for k in range(len(invalid_triangles)):
                    single_poins+=(points_in_invalid[i] in invalid_triangles[k])*(points_in_invalid[j] in invalid_triangles[k])
                if single_poins ==1:
                    delaunay.append(triangles_by_points(points_in_invalid[i], points_in_invalid[j], points[pt_num]))
    return delaunay

def triangles_to_lines(triangles:List[List]):
    output_lines = []
    for el in triangles:
        el.append(el[0])
        output_lines.append(el[:2])
        output_lines.append(el[2:])
    return output_lines