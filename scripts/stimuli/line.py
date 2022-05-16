import math
import numpy as np
from .geometry_utils import ccw_angle
from scipy.spatial.distance import euclidean

class Line:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2
        self.length = self.length()
        
    def stringify(self):
        return f"({self.point1[0], self.point1[1]}), ({self.point2[0], self.point2[1]})"

    def endpoints(self):
        return [tuple(self.point1), tuple(self.point2)]

    def length(self):
        return np.linalg.norm(self.point1-self.point2)

    def distance_to_point(self, p3):
        # Returns the minimum distance from any point on the line to p3. 0 if p3 is on the line segment.
        line = self.point2-self.point1
        # If the point is past the line segment in either direction, the closest point is the endpoint
        p2_to_dot = p3-self.point2
        if (p2_to_dot).dot(line) > 0:
            return np.linalg.norm(p2_to_dot)
        p1_to_dot = p3-self.point1
        if (p1_to_dot).dot(line) < 0:
            return np.linalg.norm(p1_to_dot)
        # Otherwise return the length of the perpendicular line to the point
        return np.linalg.norm(np.cross(line, p1_to_dot))/np.linalg.norm(line)


    def intersects(self, other_line):
        # Return true if this line intersects other_line
        # Adapted from https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

        def ccw(A,B,C):
            # Returns true if points A, B, and C are in counterclockwise order
            return (C[1]-A[1]) * (B[0]-A[0]) > (C[0]-A[0]) * (B[1]-A[1])

        if self.distance_to_point(other_line.point1) == 0 or self.distance_to_point(other_line.point1) == 0:
            return True
            
        else:
            # Only works if the lines are not parallel, hence the previous check
            return ccw(self.point1,other_line.point1,other_line.point2) != ccw(self.point2,other_line.point1,other_line.point2) and ccw(self.point1,self.point2,other_line.point1) != ccw(self.point1,self.point2,other_line.point2)    
