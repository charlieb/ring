from math import sqrt, sin, cos, radians, pi, isnan
from random import choice, random, randrange
import numpy as np
from numba import jit, int64, float64
import svgwrite as svg

X = 0
Y = 1
R = 2
default_radius = 2.

constrain_circular = 1
constrain_rect = 2
@jit((int64, float64[:,:], int64), nopython=True)
def constrain(npoints, points, mode):
    for i in range(npoints):
        p = points[i]
        if mode == constrain_circular:
            cx,cy,r = 0.,0.,100.
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

interpenetration_margin = 0.2
max_tries = 100
disable_threshold = 0.01
@jit((int64, float64[:,:]), nopython=True)
def iterate(npoints, points):
    exclusion_complete = False
    tries = 0
    while not exclusion_complete and tries < max_tries:
        tries += 1
        exclusion_complete = True

        for i1 in range(npoints):
            for i2 in range(i1+1, npoints):

                delta = points[i1][:R] - points[i2][:R]
                d = sqrt(np.sum(delta**2))
                # neighbours should be exactly 2R appart - they never will be so
                # always correct neighbours
                # non-neighbours >= 2R
                if abs(i1 - i2) == 1 or d < points[i1][R] + points[i2][R]:
                        m = d - (points[i1][R] + points[i2][R])
                        mv = (delta / d) * m / 2.
                        m = abs(m)
                        points[i1][:R] -= mv
                        points[i2][:R] += mv

                        # record movement

                        if m > interpenetration_margin:
                            exclusion_complete = False


from shapely.geometry import Point, Polygon, LinearRing, LineString

@jit(int64(int64, int64, float64[:,:]), nopython=True)
def run(iterations, npoints, points):
    jitter_size = 0.01
    
    stall = 1000
    stall_count = 0
    start = np.array([0,-90])
    
    while npoints < iterations:
        #npoints = insert(npoints, points, [random() * jitter_size, random() * jitter_size, default_radius])
        npoints = insert(npoints, points, np.array([start[X] + random() * jitter_size, 
                                           start[Y] + random() * jitter_size,
                                           default_radius]))
        print(npoints)
        stall_count = 0
        while np.sqrt(np.sum((points[npoints-1][:R] - start)**2)) < default_radius*2:

            stall_count += 1
            if stall_count >= stall:
                return npoints

            for i in range(npoints):
                points[i][Y] += 0.05

            iterate(npoints, points)

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

    iterations = 1000
    frames = 1
    for i in range(frames):
        npoints = run(iterations, npoints, points)
        draw(npoints, points, i+2)

if __name__ == '__main__':
    main()

