import math
import numpy as np
from geometry_utils import ccw_angle
from scipy.spatial.distance import euclidean

class Ellipse:
    def __init__(self, line, focus_to_edge, width):
        self.line = line
        self.width = width
        self.center = (line[0] + line[1])/2
        c = self.magnitude(line[0]-line[1])/2
        self.a = c + focus_to_edge #
        self.b = (self.a**2 - c**2)**(1/2)
        self.angle = ccw_angle(line[0], line[1], just_lower=True)
        
    def magnitude(self, line):    
        return (np.sum((line)**2))**(1/2)

    def radius_at_angle(self, angle):
        angle = abs(angle-self.angle)
        return ((self.a * self.b) / ((self.a**2) * math.sin(math.radians(angle))**2 + (self.b**2) * math.cos(math.radians(angle))**2)**(1/2)) + self.width

    def stringify(self):
        return f"Line: {self.line} \nCenter: {self.center}\nAngle: {self.angle}\na: {self.a}\nb: {self.b}"

    def transform_point(self, point):
        t = math.radians(self.angle)
        x, y = point[:,0], point[:,1]
        u = (x*math.cos(t) + y*math.sin(t))/self.a
        v = (x*math.sin(t) - y*math.cos(t))/self.b
        return np.column_stack((u,v))

    def untransform_point(self, point):
        t = math.radians(self.angle)
        u, v = point[:,0], point[:,1]
        x = self.a * u * math.cos(t) + self.b * v * math.sin(t)
        y = self.a * u * math.sin(t) - self.b * v * math.cos(t)
        return np.column_stack((x,y))

    def closest_point_to_point(self, point):
        transformed_center = self.transform_point(self.center[np.newaxis, :])[0]
        transformed_point = self.transform_point(point[np.newaxis, :])[0]
        [angle, _] = ccw_angle(transformed_center, transformed_point)
        if angle > 180:
            angle = 360 - angle
        x_change = math.cos(math.radians(angle))
        y_change = math.sin(math.radians(angle))
        if transformed_point[0] < transformed_center[0]:
            x_change *= -1
        if transformed_point[1] < transformed_center[1]:
            x_change *= -1
        transformed_closest_point = transformed_center + np.array([x_change, y_change])
        return self.untransform_point(transformed_closest_point[np.newaxis, :])[0]
    

    def intersects_ellipse(self, ellipse, ellipse_ellipse_spacing):
        my_point = self.closest_point_to_point(ellipse.center)
        his_point = ellipse.closest_point_to_point(my_point)
        return euclidean(my_point, his_point) - ellipse.width - self.width
