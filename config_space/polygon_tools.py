# Some basic tools for polygonal geometry

# Convex hull part sourced from:
#     https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain#Python
# Partially adapted from C++ code Dan Sunday, at http://geomalgorithms.com/a09-_intersect-3.html
# // Copyright 2001 softSurfer, 2012 Dan Sunday
# // This code may be freely used and modified for any purpose
# // providing that this copyright notice is included with it.
# // SoftSurfer makes no warranty for this code, and cannot be held
# // liable for any real or imagined damage resulting from its use.
# // Users of this code must verify correctness for their application.
# //
# // Assume that classes are already given for the objects:
# //    Point with 2D coordinates {float x, y;}
# //    Polygon with n vertices {int n; Point *V;} with V[n]=V[0]
# //    Tnode is a node element structure for a BBT
# //    BBT is a class for a Balanced Binary Tree
# //        such as an AVL, a 2-3, or a  red-black tree
# //        with methods given by the  placeholder code:

from collections import namedtuple

# Points should be 2-tuple (x,y)
Point = namedtuple('Point', ('x', 'y'))
OrderedEdge = namedtuple('OrderedEdge', ('lp', 'rp'))


class PointList(list):
    bounds = None

    def __init__(self, p_in):
        p_out = []
        for p in p_in:
            if not isinstance(p, Point):
                assert len(p) == 2
                p = Point(p[0], p[1])
            p_out.append(p)
        super(PointList, self).__init__(p_out)

    def min_yx_index(self):
        im = 0
        for i, pi in enumerate(self):
            if pi.y < self[im].y:
                im = i
            elif pi.y == self[im].y and pi.x < self[im].x:
                im = i
        return im

    def swap(self, i, j):
        self[i], self[j] = self[j], self[i]

    def get_bounds(self):
        # returns [minx, maxx, miny, maxy]
        if self.bounds is None:
            self.bounds = [min(self, key=lambda t:t[0])[0],
                           max(self, key=lambda t:t[0])[0],
                           min(self, key=lambda t:t[1])[1],
                           max(self, key=lambda t:t[1])[1]]
        return self.bounds

    def get_xy(self):
        x, y = zip(*self)
        return x, y


class Polygon(PointList):
    # List of points (with closure), use method edges() to get iterator over edges
    # This automatically closes the polygon, so that Polygon[n] = Polygon[0], but length (n) is still number of verts

    # def __getitem__(self, key):
    #     if key == len(self):
    #         return super(Polygon, self).__getitem__(0)
    #     else:
    #         return super(Polygon, self).__getitem__(key)
    #
    # def __iter__(self):
    #     for p in super(Polygon, self).__iter__():
    #         yield p
    #     yield self[0]
    #
    # def vertices(self):
    #     return self[:len(self)]

    def edges(self):
        for i in range(len(self)-1):
            yield self[i], self[i+1]
        yield self[-1], self[0]

    def get_edge(self, i):
        return self[i], self[(i+1) % len(self)]

    def point_inside_cn(self, p):
        #  crossing_number_poly(): crossing number test for a point in a polygon
        #       Input:   P = a point,
        #                V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
        #       Return:  0 = outside, 1 = inside
        #  This code is patterned after [Franklin, 2000] (normally just use winding method)
        cn = 0    # crossing number counter

        #  loop through all edges of the polygon
        for v0, v1 in self.edges():    # edge from V[i]  to V[i+1]
            if ((v0.y <= p.y) and (v1.y > p.y)) or ((v0.y > p.y) and (v1.y <= p.y)):    # a downward crossing
                # Compute  the actual edge-ray intersect x-coordinate
                vt = (p.y - v0.y) / (v1.y - v0.y)
                if p.x <  v0.x + vt * (v1.x - v0.x):  # P.x < intersect
                     cn += 1   # a valid crossing of y=P.y right of P.x

        return bool(cn % 2)    # 0 if even (out), and 1 if  odd (in)

    def point_inside(self, p):
        #  point_inside(): winding number test for a point in a polygon
        #       Input:   p = a point,
        #       Return:  wn = the winding number (=0 only when P is outside)

        wn = 0  # the  winding number counter

        #  loop through all edges of the polygon
        for v0, v1 in self.edges():  # edge from V[i] to  V[i+1]
            if v0.y <= p.y:  # start y <= P.y
                if v1.y > p.y:  # an upward crossing
                    if is_left(v0, v1, p) > 0:  # P left of  edge
                        wn += 1  # have  a valid up intersect

            else:  # start y > P.y (no test needed)
                if v1.y <= p.y:  # a downward crossing
                    if is_left(v0, v1, p) < 0:  # P right of  edge
                        wn -= 1  # have  a valid down intersect
        return wn

    def intersect(self, poly2):
        assert isinstance(poly2, Polygon)
        bounds1 = self.get_bounds()
        bounds2 = poly2.get_bounds()

        if (bounds2[1] <= bounds1[0] or bounds2[0] >= bounds1[1] or
                bounds2[3] <= bounds1[2] or bounds2[2] >= bounds1[3]):
            return False

        for p in poly2:
            if self.point_inside(p):
                return True
        for p in self:
            if poly2.point_inside(p):
                return True

        all_edges = []
        def add_ordered_edges(edge_list, new_edges):
            for p0, p1 in new_edges:
                if p0 < p1:
                    edge_list.append(OrderedEdge(p0, p1))
                else:
                    edge_list.append(OrderedEdge(p1, p0))

        my_edges = []
        add_ordered_edges(my_edges, self.edges())
        your_edges = []
        add_ordered_edges(your_edges, poly2.edges())
        for e1 in my_edges:
            for e2 in your_edges:
                if line_intersect(e1, e2):
                    return True
        return False

        """ Something (occasionally) wrong with my edge ordering in Shamos-Hoey booooooo~~ """
        # add_ordered_edges(all_edges, self.edges())
        # add_ordered_edges(all_edges, poly2.edges())
        #
        # event_queue = EventQueue(all_edges)
        # segment_list = []
        # def find_list_index(seglist, new_edge):
        #     i = -1
        #     for i, edge_index in enumerate(seglist):
        #         if is_left(new_edge.vertex, *all_edges[edge_index]):
        #             return i
        #     return i+1
        #
        # for e in event_queue.events:
        #
        #     if e.is_left_end:
        #         # We're adding it to the segment list
        #         # If it's a left end (new segment), check who is above, put in the list
        #         i = find_list_index(segment_list, e)
        #         segment_list.insert(i, e.edge_id)
        #         if i > 0 and line_intersect(all_edges[e.edge_id], all_edges[segment_list[i-1]]):
        #             return True
        #         if i < (len(segment_list)-1) and line_intersect(all_edges[e.edge_id], all_edges[segment_list[i+1]]):
        #             return True
        #
        #     else:
        #         # It's a right end and we need to pull it out of the queue
        #         i = segment_list.index(e.edge_id)
        #         if 0 < i < (len(segment_list)-1) and line_intersect(all_edges[segment_list[i-1]], all_edges[segment_list[i+1]]):
        #             return True
        #         segment_list.pop(i)
        #
        # return False


class Rectangle(Polygon):
    def __init__(self, xlim, ylim):
        super(Rectangle, self).__init__([[xlim[0], ylim[0]], [xlim[1], ylim[0]], [xlim[1], ylim[1]], [xlim[0], ylim[1]]])


def line_intersect(l0, l1):
    # Assume ordered lines (OrderedEdge objects)
    lsign = is_left(l0.lp, l0.rp, l1.lp)    #  l1 left point sign
    rsign = is_left(l0.lp, l0.rp, l1.rp)    #  l1 right point sign
    if (lsign * rsign >= 0):                 # l1 endpoints have same sign  relative to l0
        return False                        # => on same side => no intersect is possible
    lsign = is_left(l1.lp, l1.rp, l0.lp)    #  l0 left point sign
    rsign = is_left(l1.lp, l1.rp, l0.rp)    #  l0 right point sign
    if (lsign * rsign >= 0):                 # l0 endpoints have same sign  relative to l1
        return False                        # => on same side => no intersect is possible
    return True                             # => an intersect exists


def convex_hull(points, return_copy=False):
    """Computes the convex hull of a set of 2D points.

        Input: an iterable sequence of (x, y) pairs representing the points.
        Output: a list of vertices of the convex hull in counter-clockwise order,
          starting from the vertex with the lexicographically smallest coordinates.
        Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """
    if return_copy:
        raise NotImplementedError

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    def ccw(p1, p2, p3):
        # Three points are a counter-clockwise turn if ccw > 0, clockwise if
        # ccw < 0, and collinear if ccw = 0 because ccw is a determinant that
        # gives twice the signed  area of the triangle formed by p1, p2 and p3
        return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)

    # Build lower hull
    lower = PointList([])
    for p in points:
        while len(lower) >= 2 and ccw(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = PointList([])
    for p in reversed(points):
        while len(upper) >= 2 and ccw(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return Polygon(lower[:-1] + upper[:-1])


def xy_order(p1, p2):
    # Determines the xy lexicographical order of two points
    # Returns: (+1) if p1 > p2; (-1) if p1 < p2; and  0 if equal
    if p1.x > p2.x:
        return 1
    if p1.x < p2.x:
        return -1
    # tiebreak with y
    if p1.y > p2.y:
        return 1
    if p1.y < p2.y:
        return -1
    # otherwise same point
    return 0


def is_left(p0, p1, p2):
    # tests if point P2 is Left|On|Right of the line P0 to P1.
    # returns: >0 for left, 0 for on, and <0 for  right of the line.
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)

# ===================================================================


class Event:
    def __init__(self, edge_id, is_left_end, vertex):
        self.edge_id = edge_id
        self.is_left_end = is_left_end
        self.vertex = vertex


def event_compare(event0, event1):
    # Compare (order) two events
    return xy_order(event0.vertex, event1.vertex)


class EventQueue(object):

    def __init__(self, edge_list):
        self.events =[]
        for i, (v0, v1) in enumerate(edge_list):
            left_first = (v0 <= v1)  # (xy_order(v0, v1) >= 0)
            self.events.append(Event(i, left_first, v0))
            self.events.append(Event(i, (not left_first), v1))

        self.events = sorted(self.events, cmp=event_compare)

    def pop(self, index=-1):
        return self.events.pop(index)



class SweepLineSegment(object):

    def __init__(self, edge_id, p_left, p_right, above=None, below=None):
        self.edge_id = edge_id
        self.p_left = p_left
        self.p_right = p_right
        self.above = above
        self.below = below

    def set_above(self, above):
        self.above = above

    def set_below(self, below):
        self.below = below

class SweepLine:

    def __init__(self, poly):
        self.nv = len(poly)
        self.poly = poly
        self.tree = {}


    def add(self, event):
        v0, v1 = self.poly.get_edge(event.edge_id)
        if v0 < v1:
            new_seg = SweepLineSegment(event.edge_id, p_left=v0, p_right=v1)
        else:
            new_seg = SweepLineSegment(event.edge_id, p_left=v1, p_right=v0)
        self.tree[event.edge_id] = new_seg
