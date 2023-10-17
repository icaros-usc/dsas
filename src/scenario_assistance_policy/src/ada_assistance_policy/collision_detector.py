import numpy as np

def is_inside_sphere(p1, c, r, padding = 0):
    if np.linalg.norm(p1-c) < r+padding:
        return True
    else:
        return False


def line_sphere_intersection(p1, p2, c, r, padding = 0):
    """
    Implements the line-sphere intersection algorithm.
    https://en.wikipedia.org/wiki/Line-sphere_intersection
    :param p1: start of line segment
    :param p2: end of line segment
    :param c: sphere center
    :param r: sphere radius
    :returns: discriminant (value under the square root) of the line-sphere
        intersection formula, as a np.float64 scalar
    """
    # line
    r = r + padding 
    o = p1
    d = np.linalg.norm(p2 - p1)
    l = (p2 - p1) / d

    # discriminant
    b_square = l.dot(o - c) ** 2
    four_ac = np.linalg.norm(o - c) ** 2 - r ** 2
    discriminant = b_square - four_ac


    #from IPython import embed 
    #embed()   
    #check if both points are inside the sphere
    four_ac2 = np.linalg.norm(p2-c) ** 2 - r ** 2
    if four_ac < 0 and four_ac2 < 0:
        return [four_ac, four_ac2]
    # if discriminant is positive or zero, there is an intersection
    if discriminant < 0:
        #no intersection exists
        return []
    else: 
        #if discriminant == 0:
        #    print "one point found!"
        #there is an intersection point
        #from IPython import embed
        #embed()
        d1 = -l.dot(o-c)-np.sqrt(discriminant)
        d2 = -l.dot(o-c)+np.sqrt(discriminant)
        D = np.linalg.norm(p2-p1)
        found = []
        if (d1 >= 0 and d1 <= D):
            # the intersection is within the segment
            inter1 = o + l*d1
            found.append(inter1)

        if (d2 >= 0 and d2 <= D):
            # the intersection is within the segment
            inter2 = o + l*d2
            found.append(inter2)
        
        return found
