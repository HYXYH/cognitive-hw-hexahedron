import numpy as np
import sympy as sp
from sympy.algebras.quaternion import Quaternion
import random


class Cube:
    points = []
    edges = []
    faces = []

    projected_points = []
    sorted_faces = []

    def __init__(self, center, edge_size):
        self.center = center
        self.edge_size = edge_size
        for x in [center[0] - edge_size / 2, center[0] + edge_size / 2]:
            for y in [center[1] - edge_size / 2, center[1] + edge_size / 2]:
                for z in [center[2] - edge_size / 2, center[2] + edge_size / 2]:
                    self.points.append([x, y, z])

        for i in range(8):
            for j in range(i + 1, 8, 1):
                same_coords = 0
                same_coords += 1 if self.points[i][0] == self.points[j][0] else 0
                same_coords += 1 if self.points[i][1] == self.points[j][1] else 0
                same_coords += 1 if self.points[i][2] == self.points[j][2] else 0
                if same_coords == 2:
                    self.edges.append([i, j])

        self.faces = [
            [i for i in range(8) if self.points[i][0] == center[0] - edge_size / 2],  # fx
            [i for i in range(8) if self.points[i][0] == center[0] + edge_size / 2],
            [i for i in range(8) if self.points[i][1] == center[1] - edge_size / 2],  # fy
            [i for i in range(8) if self.points[i][1] == center[1] + edge_size / 2],
            [i for i in range(8) if self.points[i][2] == center[2] - edge_size / 2],  # fz
            [i for i in range(8) if self.points[i][2] == center[2] + edge_size / 2]
        ]
        for i in range(6):
            self.__fix_face(i)

    def __fix_face(self, face_id):
        self.faces[face_id] = [
            self.faces[face_id][0],
            self.faces[face_id][1],
            self.faces[face_id][3],
            self.faces[face_id][2]
        ]

    def move(self, direction):
        for i in range(8):
            self.points[i][0] += direction[0]
            self.points[i][1] += direction[1]
            self.points[i][2] += direction[2]

    def rotate_around_axis(self, axis, angle):
        transform_matrix = Quaternion.from_axis_angle((axis[0], axis[1], axis[2]), angle).to_rotation_matrix(
            (self.center[0], self.center[1], self.center[2]))
        for i in range(8):
            p = self.points[i]
            spp = sp.Point3D(p[0], p[1], p[2])
            spp = spp.transform(transform_matrix)
            self.points[i] = [spp.x.evalf(), spp.y.evalf(), spp.z.evalf()]

    def get_face_weight(self, face, camera_pos):
        dist = 0
        for pid in face:
            p = self.points[pid]
            dist += sp.sqrt(
                (p[0] - camera_pos[0][0]) ** 2 + (p[1] - camera_pos[1][0]) ** 2 + (p[2] - camera_pos[2][0]) ** 2)
        return dist

    def project(self, camera):
        self.projected_points = [camera.world_to_image(np.array([p]).T) for p in self.points]
        weighted_faces = [(f, self.get_face_weight(f, camera.get_camera_center())) for f in self.faces]
        weighted_faces.sort(key=lambda x: x[1], reverse=True)
        self.sorted_faces = [f[0] for f in weighted_faces]


class Hexahedron(Cube):

    def __init__(self, center, edge_size, seed):
        super().__init__(center, edge_size)
        self.max_offset = self.edge_size / 4
        self.max_angle_offset = np.pi / 8
        self.generate(seed)

    def generate(self, seed):
        random.seed(seed)

        faces2use = random.sample([0, 1, 2, 3, 4, 5], random.randint(0, 5))
        print(f"randomizing {len(faces2use)} faces")
        for f in faces2use:
            self.rotate_face(f)
            self.offset_face(f)

    def rotate_face(self, fid):
        connected_edges = [self.get_connected_edge(self.faces[fid][i], fid) for i in range(4)]

        # rotate face plane around random axis with origin at face centre
        plane = self.face2plane(fid)
        origin = self.face_centre(fid)
        axis = plane.random_point() - plane.random_point()
        transform_matrix = Quaternion.from_axis_angle((axis[0], axis[1], axis[2]),
                                                      random.uniform(-self.max_angle_offset, self.max_angle_offset)
                                                      )\
            .to_rotation_matrix((origin[0], origin[1], origin[2]))

        norm = sp.Point3D(plane.normal_vector)
        plane = sp.Plane(origin, normal_vector=norm.transform(transform_matrix))

        # find new points for face
        new_face_points = []
        for ce in connected_edges:
            line = sp.Line3D(sp.Point3D(self.points[ce[0]]), sp.Point3D(self.points[ce[1]]))
            new_point = plane.intersection(line)[0]
            new_point = [new_point.x.evalf(), new_point.y.evalf(), new_point.z.evalf()]
            new_face_points.append(new_point)

            if not self.is_b_not_far_than_a(self.points[ce[0]], self.points[ce[1]], new_point):
                print("skip face rotating - hex will not be valid")
                return

        # replace old points with new
        for i in range(4):
            pid = self.faces[fid][i]
            self.points[pid] = new_face_points[i]

    def offset_face(self, fid):
        connected_edges = [self.get_connected_edge(self.faces[fid][i], fid) for i in range(4)]

        plane = self.face2plane(fid)
        origin = self.face_centre(fid)
        norm = sp.Point3D(plane.normal_vector)
        offset_vector = (sp.Point3D(self.center) - origin)
        offset_vector /= sp.Segment3D(sp.Point3D(0, 0, 0), offset_vector).length
        origin += offset_vector * random.uniform(-self.max_offset, self.max_offset)
        plane = sp.Plane(origin, normal_vector=norm)

        # find new points for face
        new_face_points = []
        for ce in connected_edges:
            line = sp.Line3D(sp.Point3D(self.points[ce[0]]), sp.Point3D(self.points[ce[1]]))
            intersections = plane.intersection(line)
            if len(intersections) == 0:
                print("skip face offsetting - hex will not be valid")
                return

            new_point = intersections[0]
            new_point = [new_point.x.evalf(), new_point.y.evalf(), new_point.z.evalf()]
            new_face_points.append(new_point)

            if not self.is_b_not_far_than_a(self.points[ce[0]], self.points[ce[1]], new_point):
                print("skip face offsetting - hex will not be valid")
                return

        # replace old points with new
        for i in range(4):
            pid = self.faces[fid][i]
            self.points[pid] = new_face_points[i]

    def is_b_not_far_than_a(self, origin, a, b):
        origin = sp.Point3D(origin[0], origin[1], origin[2])
        a = sp.Point3D(a[0], a[1], a[2])
        b = sp.Point3D(b[0], b[1], b[2])

        v1 = a - origin
        v2 = b - origin
        dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
        if dot < 0:
            return True
        return sp.Segment3D(origin, a).length > sp.Segment3D(origin, b).length

    def face2plane(self, fid):
        p1 = sp.Point3D(self.points[self.faces[fid][0]])
        p2 = sp.Point3D(self.points[self.faces[fid][1]])
        p3 = sp.Point3D(self.points[self.faces[fid][2]])
        return sp.Plane(p1, p2, p3)

    def face_centre(self, fid):
        p1 = sp.Point3D(self.points[self.faces[fid][0]])
        p2 = sp.Point3D(self.points[self.faces[fid][1]])
        p3 = sp.Point3D(self.points[self.faces[fid][2]])
        p4 = sp.Point3D(self.points[self.faces[fid][2]])
        return (p1 + p2 + p3 + p4) / 4

    # возвращает рёбро, которое соединено с гранью fid в точке pid
    def get_connected_edge(self, pid, fid):
        face = self.faces[fid]
        for e in self.edges:
            if pid not in e:
                continue
            p2 = e[0] if e[0] != pid else e[1]
            if p2 in face:
                continue
            return [pid, p2]
        raise Exception("connected edge not found!")
