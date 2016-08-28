from math import sqrt, sin, cos, radians, pi, isnan
from random import choice, random, randrange
import numpy as np
from numba import jit, int64, float64
import svgwrite

#@jit((float64, int64, float64[:,:], int64[:,:]))
def ring(radius, point_spacing, points, connections):
    npoints = int(2*pi*radius / point_spacing)
    for i in range(npoints):
        a = i * 2 * pi / npoints
        points[i][0] = radius*cos(a)
        points[i][1] = radius*sin(a)
        connections[i][0] = (i - 1) % npoints
        connections[i][1] = (i + 1) % npoints
    return npoints

#@jit(int64(float64[:,:], int64[:,:]))
def setup(points, connections, point_spacing, nrings, mode='con'):
    '''Mode can be concentric ('con') or adjacent ('adj')'''
    starting_points = []
    npoints = 0

    radius = 10
    
    for i in range(nrings):
        starting_points.append(npoints)

        added_points = ring(radius * (i+1 if mode =='con' else 1),
                            point_spacing,
                            points[npoints:],
                            connections[npoints:])
        if mode == 'adj':
            points[npoints:npoints+added_points] += (radius + 5) * (i-nrings / 2)

        connections[npoints:npoints+added_points] += npoints
        npoints += added_points

    return npoints, starting_points
    
constrain_circular = 1
constrain_rect = 2

@jit((int64, float64[:,:], int64, float64[:]))
def constrain(npoints, points, mode, constrain_data):
    for i in range(npoints):
        p = points[i]
        if mode == constrain_circular:
            cx,cy,r = constrain_data[0], constrain_data[1], constrain_data[2]
            dx = p[0] - cx
            dy = p[1] - cy
            dmag = dx*dx + dy*dy 
            if (dmag > r*r):
                dmag = sqrt(dmag)
                p[0] = cx + r * dx / dmag
                p[1] = cy + r * dy / dmag


@jit(int64(int64[:], int64, float64[:,:], int64[:,:]))
def insert(starting_points, npoints, points, connections):
    for start in starting_points:
        visited = [start]
        n = connections[start][1]
        while n not in visited:
            visited.append(n)
            n = connections[n][1]

        n1 = choice(visited)
        p1 = points[n1]
        c1 = connections[n1]

        n2 = c1[1]
        p2 = points[n2]
        c2 = connections[n2]

        cnew = connections[npoints]
        
        points[npoints] = p1 + (p2 - p1) / 2
        
        #print(n1, c1, n2, c2)
        c1[1] = npoints
        cnew[0] = n1
        cnew[1] = n2
        c2[0] = npoints
        #print(n1, c1, n2, c2, cnew)

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

@jit(int64(int64, int64, int64[:], float64, float64, float64[:,:], int64[:,:]))
def run(iterations, npoints, starting_points, point_spacing, ring_spacing, points, connections):
    insert_rate = 1
    attr_dist = point_spacing
    repel_dist = ring_spacing

    n = insert_rate
    for _ in range(iterations):
        n -= 1
        if n == 0:
            npoints = insert(starting_points, npoints, points, connections)
            n = insert_rate

        iterate(npoints, points, connections, attr_dist, repel_dist)

    minx = miny =  9999999
    maxx = maxy = -9999999
    for i in range(npoints):
        minx = points[i][0] if points[i][0] < minx else minx
        maxx = points[i][0] if points[i][0] > maxx else maxx
        miny = points[i][1] if points[i][1] < miny else miny
        maxy = points[i][1] if points[i][1] > maxy else maxy

    constrain = False
    if constrain:
        const_iterations = 10.
        cx = (maxx + minx) / 2
        cy = (maxy + miny) / 2
        minr = min((maxx - minx), (maxy - miny)) / 2 
        maxr = max((maxx - minx), (maxy - miny)) / 2 
        for i in range(const_iterations):
            constrain(npoints, points, constrain_circular, 
                    np.array([cx, cy,
                             maxr - ((maxr - minr) + ring_spacing)*i/const_iterations]))

            for _ in range(const_iterations):
                iterate(npoints, points, connections, attr_dist, repel_dist)

    return npoints

def walk(start, connections):
    visited = []
    n = start
    yield(n)
    while n not in visited:
        visited.append(n)
        n = connections[n][1]
        yield(n)

def walk_coords(start, points, connections):
  return ((points[n][0], points[n][1]) for n in walk(start, connections))

def path(start, points, connections):
  return ((('M',) if i == 0 else ()) + p for i,p in enumerate(walk_coords(start, points, connections)))

def draw(starting_points, npoints, points, connections, frame):
    dwg = svgwrite.Drawing('test%05i.svg'%frame, profile='tiny')
    minx = miny =  9999999
    maxx = maxy = -9999999
    line_width = 1
    for i in range(npoints):
        minx = points[i][0] if points[i][0] < minx else minx
        maxx = points[i][0] if points[i][0] > maxx else maxx
        miny = points[i][1] if points[i][1] < miny else miny
        maxy = points[i][1] if points[i][1] > maxy else maxy
    dwg.viewbox(minx=minx-line_width, miny=miny-line_width, 
                width=maxx-minx+2*line_width, height=maxy-miny+2*line_width)
    # reverse starting points so that if concentric the largest one
    # will be first so that when they're drawn with fill they'll all be 
    # visible
    rev_starts = starting_points.copy()
    rev_starts.reverse()
    #fills = ['green', 'blue', 'red'] * len(starting_points) # far too long but meh
    fills = ['none'] * len(starting_points)
    for start, fill in zip(rev_starts, fills):
        svgpath = svgwrite.path.Path(path(start, points, connections), fill=fill)
        svgpath.push('Z')
        svgpath.stroke('black', width=1)
        dwg.add(svgpath)
    dwg.save()

def main():
    array_len = 5000
    points = np.zeros((array_len,2), dtype=np.float64)
    connections = np.zeros((array_len,2), dtype=np.int64)

    point_spacing = 2.
    ring_spacing = 3.
    npoints, starting_points = setup(points, connections, point_spacing, 3, mode='con')

    iterations = 200
    frames = 1
    for i in range(frames):
        npoints = run(iterations, npoints, starting_points, point_spacing, ring_spacing, points, connections)
        draw(starting_points, npoints, points, connections, i)

if __name__ == '__main__':
    main()

