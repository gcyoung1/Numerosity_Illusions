import math
import numpy as np
from geometry_utils import ccw_angle
from scipy.spatial.distance import euclidean

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        
    def stringify(self):
        return f"Center: {self.center}\nRadius: {self.radius}"

    def distance_from(self, other_circle):
        center_center_distance = euclidean(self.center, other_circle.center)
        return center_center_distance - self.radius - other_circle.radius

    def corners(self):
        corners = [self.center-self.radius,self.center-self.radius,self.center+self.radius,self.center+self.radius]