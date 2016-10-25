from math import sqrt, sin, cos, radians, pi, isnan, atan2
from random import choice, random, randrange
import numpy as np
from numba import jit, int64, float64, boolean
import svgwrite as svg

X = 0
Y = 1
R = 2
default_radius = 2.

def random_from(pt, new_pt):
    # random - random is most likely to be 0 and falls off toward +-1
    angle = (random() - random()) * pi
    angle_to_origin = pi + atan2(pt[Y], pt[X])
    new_pt[X] = pt[X] + (new_pt[R] + pt[R]) * cos(angle_to_origin + angle)
    new_pt[Y] = pt[Y] + (new_pt[R] + pt[R]) * sin(angle_to_origin + angle)

margin = 0.01
max_tries = 10
@jit(boolean(int64, float64[:]))
def add_point(n, points):
    new_hit = True
    tries = 0
    while new_hit and tries < max_tries:
        tries += 1
        random_from(points[n -1], points[n])

        new_hit = False
        for i in range(n):
            dist = np.sqrt(np.sum((points[i][:R] - points[n][:R])**2))
            if dist + margin < points[i][R] + points[n][R]:
                new_hit = True
                break

    return tries < max_tries

np.set_printoptions(linewidth=1000)
@jit((int64, float64[:]))
def search(max_points, points):
    npoints = 1
    points[:,R] = default_radius
    points[0][:R] = [0.,0.]
    tries = np.array([0] * max_points, dtype='int64')

    iterations = 0

    while npoints < max_points:
        iterations += 1
        if add_point(npoints, points):
            npoints += 1
        else:
            npoints -= 1
            tries[npoints] += 1
            # backtrack
            momentum = 1
            while tries[npoints] > max_tries or momentum > 0:
                if tries[npoints] > max_tries:
                    momentum += 1
                else:
                    momentum -= 1
                tries[npoints] = 0
                npoints -= 1
                tries[npoints] += 1
                
        if npoints % 10 == 0:
            print(npoints)
    print(iterations)

def draw(npoints, points, frame):
    dwg = svg.Drawing('search%05i.svg'%frame, profile='tiny')
    minx = miny =  9999999
    maxx = maxy = -9999999
    maxr = 0
    line_width = 1
    svgpath = svg.path.Path(fill='none')
    for i in range(npoints):
        minx = points[i][0] if points[i][0] < minx else minx
        maxx = points[i][0] if points[i][0] > maxx else maxx
        miny = points[i][1] if points[i][1] < miny else miny
        maxy = points[i][1] if points[i][1] > maxy else maxy
        maxr = points[i][R] if points[i][R] > maxr else maxr

        svgpath.push((('M',) if i == 0 else ()) + (points[i][X], points[i][Y]))
        
        c = svg.shapes.Circle((points[i][X], points[i][Y]), points[i][R],
                                    fill='none', 
                                    stroke='blue',
                                    stroke_width=line_width)
        dwg.add(c)

    c = svg.shapes.Circle((0,0), 5,
                                fill='none', 
                                stroke='red',
                                stroke_width=line_width)
    dwg.add(c)

#    dwg.viewbox(minx=minx-line_width-maxr, miny=miny-line_width-maxr, 
#                width=maxx-minx+2*line_width+maxr, height=maxy-miny+2*line_width+maxr)
    dwg.viewbox(minx=-100-default_radius-line_width-1, miny =-100-default_radius-line_width-1,
                width=200+default_radius*2+line_width+1, height=200+default_radius*2+line_width+1)
    
    svgpath.stroke('black', width=1)
    dwg.add(svgpath)
    dwg.save()

def main():
    array_len = 1000
    points = np.zeros((array_len,3), dtype=np.float64)

    npoints = 1000
    search(npoints, points)
    draw(npoints, points, 2)

if __name__ == '__main__':
    main()

