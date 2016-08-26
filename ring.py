from math import sqrt, sin, cos, radians, pi, isnan
from random import choice, random, randrange
import tkinter as tk
import numpy as np
from numba import jit, int64, float64
import svgwrite

#@jit((float64, int64, float64[:,:], int64[:,:]))
def ring(radius, npoints, points, connections, x, y):
    for i in range(npoints):
        a = i * 2 * pi / npoints
        points[i][0] = radius*cos(a) + x
        points[i][1] = radius*sin(a) + y
        connections[i][0] = (i - 1) % npoints
        connections[i][1] = (i + 1) % npoints

#@jit(int64(float64[:,:], int64[:,:]))
def setup(points, connections):
    starting_points = []
    npoints = 0

    starting_points.append(npoints)
    ring(10, 10, points[npoints:npoints+10], connections[npoints:npoints+10], -10, 0)
    npoints += 10

    starting_points.append(npoints)
    ring(10, 10, points[npoints:npoints+10], connections[npoints:npoints+10], 10, 0)
    connections[npoints:npoints+10] += npoints
    npoints += 10

    #starting_points.append(npoints)
    #ring(10, 10, points[npoints:npoints+10], connections[npoints:npoints+10])
    #npoints += 10

    return npoints, starting_points
    
@jit(int64(int64, float64[:,:], int64[:,:]))
def insert(npoints, points, connections):
    n1 = randrange(npoints)
    p1 = points[n1]
    c1 = connections[n1]

    n2 = c1[0]
    p2 = points[n2]
    c2 = connections[n2]

    cnew = connections[npoints]
    
    points[npoints] = p1 + (p2 - p1) / 2
    
    #print(n2, c2, n1, c1)
    c1[0] = npoints
    cnew[1] = n1
    cnew[0] = n2
    c2[1] = npoints
    #print(n2, c2, n1, c1, cnew)

    npoints += 1
    return npoints

@jit((int64, float64[:,:], int64[:,:], float64, float64))
def iterate(npoints, points, connections, attr_dist, repel_dist):
    repel_acc = np.array([0,0], dtype=np.float64)
    attr_acc = np.array([0,0], dtype=np.float64)
    for i1 in range(npoints):
        repel_acc[0] = repel_acc[1] = 0.0
        attr_acc[0] = attr_acc[1] = 0.0
        for i2 in range(npoints):
            if i1 == i2: continue
            delta = points[i1] - points[i2]
            d = sqrt(delta[0]*delta[0] + delta[1]*delta[1])
            if i2 == connections[i1][0] or i2 == connections[i1][1]:
                # neighbour
                if d > attr_dist:
                    attr_acc -= delta / d
            else: # not neighbour
                if d < repel_dist:
                    repel_acc += delta / d

        if repel_acc[0] + repel_acc[1] != 0.0:
            repel_acc /= np.sqrt(repel_acc[0]*repel_acc[0] + repel_acc[1]*repel_acc[1])
            
        if attr_acc[0] + attr_acc[1] != 0.0:
            attr_acc /= np.sqrt(attr_acc[0]*attr_acc[0] + attr_acc[1]*attr_acc[1])

        points[i1] += 0.5 * (attr_acc + repel_acc)

@jit(int64(int64, int64, float64[:,:], int64[:,:]))
def run(iterations, npoints, points, connections):
    insert_rate = 1
    attr_dist = 2.
    repel_dist = 4.

    n = insert_rate
    for _ in range(iterations):
        n -= 1
        if n == 0:
            npoints = insert(npoints, points, connections)
            n = insert_rate

        iterate(npoints, points, connections, attr_dist, repel_dist)
    return npoints

def walk(start, connections):
    visited = []
    n = start
    yield(n)
    while n not in visited:
        visited.append(n)
        n = connections[n][0]
        yield(n)

def walk_coords(start, points, connections):
  return ((points[n][0], points[n][1]) for n in walk(start, connections))

def path(start, points, connections):
  return ((('M',) if i == 0 else ()) + p for i,p in enumerate(walk_coords(start, points, connections)))

def draw(starting_points, points, connections, frame):
    dwg = svgwrite.Drawing('test%05i.svg'%frame, profile='tiny')
    dwg.viewbox(minx=-100, miny=-100, width=200, height=200)
    fills = ['none', 'blue']
    for start, fill in zip(starting_points, fills):
        svgpath = svgwrite.path.Path(path(start, points, connections), fill=fill)
        svgpath.push('Z')
        svgpath.stroke('black', width=1)
        dwg.add(svgpath)
    dwg.save()

def main():
    array_len = 5000
    points = np.zeros((array_len,2), dtype=np.float64)
    connections = np.zeros((array_len,2), dtype=np.int64)
    npoints, starting_points = setup(points, connections)
    print(starting_points)
    print(connections[0:npoints])

    iterations = 500
    frames = 1
    for i in range(frames):
        npoints = run(iterations, npoints, points, connections)
        draw(starting_points, points, connections, i)

if __name__ == '__main__':
    main()



