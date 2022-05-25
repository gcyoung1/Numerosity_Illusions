import math
import numpy as np
from .geometry_utils import ccw_angle
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
        # Since (0,0,0,0) is a 1-pixel circle in PIL
        corners = (self.center[0]-self.radius+0.5,self.center[1]-self.radius+0.5,self.center[0]+self.radius-0.5,self.center[1]+self.radius-0.5)
        return corners
