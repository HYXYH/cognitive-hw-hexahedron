import numpy as np
import sympy as sp

def predict_direction(coords1, coords2, K):

    p11 = coords1[0].T[0]
    p12 = coords2[0].T[0]
    line1 = sp.Line2D(sp.Point2D(p11[0], p11[1]), sp.Point2D(p12[0], p12[1]))

    p21 = coords1[1].T[0]
    p22 = coords2[1].T[0]
    line2 = sp.Line2D(sp.Point2D(p21[0], p21[1]), sp.Point2D(p22[0], p22[1]))

    p = line1.intersection(line2)[0]

    k_inv = np.linalg.inv(K)

    line = np.dot(k_inv, np.array([[p.x.evalf()], [p.y.evalf()], [1]]))
    return line.T[0]
