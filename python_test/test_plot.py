# This code is to testify the 2D case of Result 3.6 shown in Fig. 3.6 in the book "Multiple View Geometry".

import numpy as np
import matplotlib.pyplot as plt
import random


def transform(p0, p1, rad, t0, t1):
    # Given 2D point (p0, p1), transform it by rotation (around (0,0,1) CCW with rad) and translation (t0, t1)
    cos = np.cos(rad)
    sin = np.sin(rad)
    x0 = cos * p0 - sin * p1 + t0
    x1 = sin * p0 + cos * p1 + t1
    return [x0, x1]


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    t0 = 2.0
    t1 = 1.0
    theta = random.uniform(10.0, 180.0)
    rad = theta * np.pi / 180.0

    cnt = 10
    for i in range(cnt):
        p0 = random.uniform(0.0, 10.0)
        p1 = random.uniform(0.0, 10.0)

        [pp0, pp1] = transform(p0, p1, rad, t0, t1)
        c0 = (p0 + pp0) / 2.0
        c1 = (p1 + pp1) / 2.0
        plt.plot(p0, p1, 'ko', markersize=10)
        plt.plot(pp0, pp1, 'ko', markersize=10)
        plt.plot(c0, c1, 'bo', markersize=10)
        plt.plot([p0, pp0], [p1, pp1], color='green')

        a0 = pp0 - p0
        a1 = pp1 - p1
        alen = np.sqrt(a0 * a0 + a1 * a1)
        m = alen / (2.0 * np.tan(rad / 2.0))
        x0 = 0
        x1 = 0
        if np.abs(pp1 - p1) > 1e-5:
            slope = (p0 - pp0) / (pp1 - p1)
            right = m * (pp1 - p1) / alen
            x0 = c0 - right
            x1 = slope * (x0 - c0) + c1
            plt.plot(x0, x1, 'ro', markersize=10)
        else:
            x0 = c0
            x1 = c1 - m
            plt.plot(x0, x1, 'ro', markersize=10)
        plt.plot([p0, x0, pp0], [p1, x1, pp1], color='red')
        plt.plot([x0, c0], [x1, c1], color='red')
    plt.show()


if __name__ == "__main__":
    main()
