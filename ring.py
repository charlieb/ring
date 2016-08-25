from math import sqrt, sin, cos, radians, pi
from random import choice, random, randrange
import tkinter as tk
import numpy as np
from numba import jitclass, int64, float64


spec = [('npoints', int64),
        ('points', float64[:,:]),
        ('array_len', int64),
        ('insert_counter', int64),
        ('connections', int64[:,:]),
        ('repel_dist', float64),
        ('attr_dist', float64)
        ]
@jitclass(spec)
class SoftRingAnimator():
    def __init__(self, npoints, array_len):
        self.points = np.zeros((array_len,2), dtype=np.float64)
        self.npoints = npoints
        self.array_len = array_len
        self.insert_counter = 0
        self.connections = np.zeros((self.array_len,2), dtype=np.int64)
        self._ring(5*self.npoints / (2*pi), 300, 300)
        self.repel_dist = 10
        self.attr_dist = 2

    def _ring(self, radius, xpos, ypos):
        angles = []
        for i in range(self.npoints):
            angles.append(i * 2 * pi / self.npoints)
        for i,a in enumerate(angles):
            self.points[i][0] = xpos + radius*cos(a)
            self.points[i][1] = ypos + radius*sin(a)
            self.connections[i][0] = (i - 1) % self.npoints
            self.connections[i][1] = (i + 1) % self.npoints
            
    def insert(self):
        if self.npoints >= self.array_len:
            return

        n1 = randrange(self.npoints)
        p1 = self.points[n1]
        c1 = self.connections[n1]

        n2 = c1[0]
        p2 = self.points[n2]
        c2 = self.connections[n2]

        cnew = self.connections[self.npoints]
        
        self.points[self.npoints] = p1 + (p2 - p1) / 2
        
        #print(n2, c2, n1, c1)
        c1[0] = self.npoints
        cnew[1] = n1
        cnew[0] = n2
        c2[1] = self.npoints
        #print(n2, c2, n1, c1, cnew)

        self.npoints += 1

    def walk(self):
        visited = []
        n = 0
        while n not in visited:
            visited.append(n)
            n = self.connections[n][0]
        print("Visited %i / %i"%(len(visited), self.npoints))

    def iterate(self):
        self.insert_counter += 1
        if self.insert_counter == 10:
            self.insert()
            self.insert_counter = 0

        repel_acc = np.array([0,0], dtype=np.float64)
        attr_acc = np.array([0,0], dtype=np.float64)
        for i1 in range(self.npoints):
            repel_acc[0] = repel_acc[1] = 0.0
            attr_acc[0] = attr_acc[1] = 0.0
            for i2 in range(self.npoints):
                if i1 == i2: continue
                delta = self.points[i1] - self.points[i2]
                d = sqrt(delta[0]*delta[0] + delta[1]*delta[1])
                if i2 == self.connections[i1][0] or i2 == self.connections[i1][1]:
                    # neighbour
                    if d > self.attr_dist:
                        attr_acc -= delta / d
                else: # not neighbour
                    if d < self.repel_dist:
                        repel_acc += delta / d

            if repel_acc[0] + repel_acc[1] != 0.0:
                repel_acc /= np.sqrt(repel_acc[0]*repel_acc[0] + repel_acc[1]*repel_acc[1])
                
            if attr_acc[0] + attr_acc[1] != 0.0:
                attr_acc /= np.sqrt(attr_acc[0]*attr_acc[0] + attr_acc[1]*attr_acc[1])
                repel_acc[0] = repel_acc[1] = 0.0

            self.points[i1] += 0.05 * (attr_acc + repel_acc)


class UI:
    def __init__(self, animator):

        self.animator = animator
        master = tk.Tk()

        self.w = tk.Canvas(master, width=600, height=600)
        self.w.pack()

        self.w.create_line(0, 0, 200, 100)
        self.w.create_line(0, 100, 200, 0, fill="red", dash=(4, 4))

        self.w.create_rectangle(50, 25, 150, 75, fill="blue")
        
        self.w.after(0,lambda: self.update(100))

        tk.mainloop()
        
    def update(self, iterations):
        for _ in range(iterations):
            self.animator.iterate()
        self.draw()
        self.w.after(1, lambda: self.update(iterations))

    def draw(self):
        self.w.delete(tk.ALL)
        print(self.animator.npoints)
        for p in self.animator.points:
            self.w.create_oval(p[0]-1, p[1]-1,
                               p[0]+1, p[1]+1, 
                               fill='black')


if __name__ == '__main__':
    UI(SoftRingAnimator(10, 5000))

