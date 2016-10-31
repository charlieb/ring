from math import sqrt, sin, cos, radians, pi, isnan
from random import choice, random, randrange
import numpy as np
from numba import jit, int64, float64
import svgwrite as svg

X = 0
Y = 1
R = 2
default_radius = 2.

DEBUG = True

constrain_circular = 1
constrain_circular_radius = 100.
constrain_rect = 2
@jit((int64, float64[:,:], int64), nopython=True)
def constrain(npoints, points, mode):
    for i in range(npoints):
        p = points[i]
        if mode == constrain_circular:
            cx,cy,r = 0.,0.,constrain_circular_radius
            dx = p[X] - cx
            dy = p[Y] - cy
            dmag = dx*dx + dy*dy 
            if (dmag > r*r):
                dmag = sqrt(dmag)
                p[X] = cx + r * dx / dmag
                p[Y] = cy + r * dy / dmag

@jit(int64(int64, float64[:,:], float64[:]), nopython=True)
def insert(npoints, points, starting_coords):
    points[npoints] = starting_coords
    npoints += 1
    return npoints

@jit(int64[:](float64[:], int64[:,:,:], float64[:], float64), nopython=True)
def get_box(point, boxes, pos_min, box_size):
    p = (point[:R] - pos_min) / box_size
    return boxes[int(p[X]), int(p[Y])]

max_tries = 100
interpenetration_margin = 0.2
@jit((int64, float64[:,:], int64[:,:,:], float64[:], float64), nopython=True)
def iterate2(npoints, points, boxes, pos_min, box_size):
    exclusion_complete = False
    tries = 0
    while not exclusion_complete and tries < max_tries:
        tries += 1
        exclusion_complete = True

        for i in range(npoints):
            p1 = points[i]
            box = get_box(p1, boxes, pos_min, box_size)
            b = 0
            while box[b] != -1:
                if box[b] == i:
                    b += 1
                    continue
                p2 = points[box[b]]

                delta = p1[:R] - p2[:R]
                d = sqrt(np.sum(delta**2))
                # neighbours should be exactly 2R appart - they never will be so
                # always correct neighbours
                # non-neighbours >= 2R
                if abs(i - box[b]) == 1 or d < p1[R] + p2[R]:
                        m = d - (p1[R] + p2[R])
                        mv = (delta / d) * m / 2.
                        m = abs(m)
                        p1[:R] -= mv
                        p2[:R] += mv

                        if m > interpenetration_margin:
                            exclusion_complete = False
                b += 1

def box_array(pos_min, pos_max, size):
    dim = (pos_max - pos_min) / size
    boxes = np.zeros([int(dim[X]), int(dim[Y]), 50], dtype='int64')
    return boxes

@jit((int64[:,:,:], float64[:], float64, int64, int64, float64[:,:]), nopython=True)
def fill_boxes(boxes, pos_min, size, box_range, npoints, points):
    for i in range(npoints):
        p = (points[i][:R] - pos_min) / size
        for x in range(int(p[X]) - box_range, int(p[X]) + box_range + 1):
            for y in range(int(p[Y]) - box_range, int(p[Y]) + box_range + 1):
                box = boxes[x,y]
                b = 0
                while box[b] != -1:
                    if b >= boxes.shape[2]:
                        print("ERROR - box too small at ",x,y, "for point", i, "at",points[i][X], points[i][Y])
                        for j in range(boxes.shape[2]):
                            print(j, box[j])
                        return
                    b += 1
                box[b] = i

from shapely.geometry import Point, Polygon, LinearRing, LineString

@jit(int64(int64, int64, float64[:,:]))
def run(iterations, npoints, points):
    jitter_size = 0.01
    
    stall = 1000
    stall_count = 0
    start = np.array([0,-90])

    box_range = 1
    box_size = default_radius * 2.5
    pos_min = np.array([-constrain_circular_radius, -constrain_circular_radius])
    boxes = box_array(pos_min, -pos_min, box_size)
    
    while npoints < iterations:
        #npoints = insert(npoints, points, [random() * jitter_size, random() * jitter_size, default_radius])
        npoints = insert(npoints, points, np.array([start[X] + random() * jitter_size, 
                                           start[Y] + random() * jitter_size,
                                           default_radius]))

            
        if npoints > 1:
            for y in range(boxes.shape[Y]):
                for x in range(boxes.shape[X]):
                    nb = 0
                    while boxes[x][y][nb] != -1: nb += 1
                    #print('.' if boxes[x][y][0] == -1 else 'o', end='')
                    print('%02.d'%nb, end='')
                print('')

            if DEBUG:
                for y in range(boxes.shape[Y]):
                    for x in range(boxes.shape[X]):
                        nb = 0
                        while boxes[x][y][nb] != -1:
                            print(boxes[x][y][nb], end=',')
                            nb += 1
                        if nb > 0:
                            print('')
                for i in range(1, npoints):
                    print(i, np.sqrt(np.sum((points[i][:R] - points[i-1][:R])**2)))

        print(npoints)

        stall_count = 0
        while np.sqrt(np.sum((points[npoints-1][:R] - start)**2)) < default_radius*2:

            stall_count += 1
            if stall_count >= stall:
                return npoints

            for i in range(npoints):
                points[i][Y] += 0.05


            boxes.fill(-1)
            fill_boxes(boxes, pos_min, box_size, box_range, npoints, points)

            #iterate(npoints, points)
            iterate2(npoints, points, boxes, pos_min, box_size)

            constrain_on = True
            if constrain_on:
                const_iterations = 1
                cx = 0.
                cy = 0.
                for i in range(const_iterations):
                    constrain(npoints, points, constrain_circular)

#            if npoints > 2 and not LineString(points[:npoints,:R]).is_simple:
#                return npoints


    return npoints

def draw(npoints, points, frame):
    dwg = svg.Drawing('test%05i.svg'%frame, profile='tiny')
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

    c = svg.shapes.Circle((0,0), 100,
                                fill='none', 
                                stroke='red',
                                stroke_width=0.5)
    dwg.add(c)

#    dwg.viewbox(minx=minx-line_width-maxr, miny=miny-line_width-maxr, 
#                width=maxx-minx+2*line_width+maxr, height=maxy-miny+2*line_width+maxr)
    dwg.viewbox(minx=-100-default_radius-line_width-1, miny =-100-default_radius-line_width-1,
                width=200+default_radius*2+line_width+1, height=200+default_radius*2+line_width+1)
    
    svgpath.stroke('black', width=1)
    dwg.add(svgpath)
    dwg.save()

def main():
    array_len = 5000
    points = np.zeros((array_len,3), dtype=np.float64)
    npoints = 0

    iterations = 2000
    frames = 1
    for i in range(frames):
        npoints = run(iterations, npoints, points)
        draw(npoints, points, i+4)

if __name__ == '__main__':
    main()

