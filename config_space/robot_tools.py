import numpy as np
import polygon_tools as poly

class Robot2D(object):

    def __init__(self, pos=poly.Point(0.0, 0.0), heading=0.0, footprint=poly.PointList([poly.Point(0.0, 0.0)])):
        self.R = np.eye(2)

        self.position = pos
        self.footprint = footprint
        self.heading = heading
        self._set_heading_transformation()

    def _set_heading_transformation(self):
        ct, st = np.cos(self.heading), np.sin(self.heading)
        self.R = np.array([[ct, -st], [st, ct]])

    def set_heading(self, heading):
        self.heading = heading
        self._set_heading_transformation()

    def set_position(self, pos):
        self.position = pos

    def set_footprint(self, footprint):
        self.footprint = footprint

    def get_current_polygon(self):
        out_poly = poly.Polygon([poly.Point(*(np.matmul(self.R, p)+self.position)) for p in self.footprint])
        return out_poly


class RobotArm2D(object):
    _spine_pts = None

    def __init__(self, base_position=[0.0, 0.0], link_lengths=[1.0, 1.0], link_angles=[0.0, 0.0]):
        # Assume arm angles are relative (can be summed)

        self._base_position = poly.Point(base_position[0], base_position[1])

        assert len(link_lengths) == len(link_angles)
        self._link_lengths = np.array(link_lengths)
        self._link_angles = np.array(link_angles)

        self._R = [np.eye(2) for i in self._link_angles]
        self._set_rotation_transforms()

    def set_link_angles(self, link_angles):
        self._link_angles = np.array(link_angles)
        self._set_rotation_transforms()

    def _set_rotation_transforms(self):
        sum_angles = self._link_angles.cumsum()
        for i, theta in enumerate(sum_angles):
            ct, st = np.cos(theta), np.sin(theta)
            self._R[i] = np.array([[ct, -st], [st, ct]])
        self._set_spine_points()

    def get_current_polygon(self):
        # Run backwards through the points to make a polygon
        return poly.Polygon(self._spine_pts + self._spine_pts[-2:0:-1])

    def _set_spine_points(self):
        self._spine_pts = [self._base_position]
        for R, ll in zip(self._R, self._link_lengths):
            self._spine_pts.append(poly.Point(*(np.matmul(R, [ll, 0])+self._spine_pts[-1])))

    def get_spine_points(self):
        return [p.x for p in self._spine_pts], [p.y for p in self._spine_pts]

    def get_end_effector_position(self):
        return self._spine_pts[-1]









