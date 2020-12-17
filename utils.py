import numpy as np
import math

def normalize(v):
    normalized_v = v / np.sqrt(np.sum(v**2))
    return normalized_v

def camera_transformation(target, camera):
    T = np.array(target)
    C = np.array(camera)
    up_vec = np.array((0, 0, 1))
    w = (C - T)/math.sqrt(np.sum((C - T) ** 2))
    u = np.cross(up_vec, w)
    v = np.cross(w, u)
    camera_mat = (np.transpose(np.stack((u, v, w, C))))
    camera_mat = np.concatenate((camera_mat, np.expand_dims(np.array((0, 0, 0, 1)), axis = 0)))
    view_mat = np.linalg.inv(camera_mat)
    return (camera_mat, view_mat)

# T = [1,-1,0]
# C = [-3,-3,4]
# print(camera_transformation(T, C))

def reflection(r, n):
    R = np.array(r)
    N = np.array(n)
    return (2 * np.dot(np.dot(n, r), n) - r)
# R = [1, 1, 0]
# N = [0 ,1, 0]
# print(reflection(R, N))

def intersection_ray_sphere(S, R, C, ray):
    S = np.array(S)
    C = np.array(C)
    ray = np.array(ray)
    V1 = S - C
    angle = math.acos(np.dot(normalize(V1), ray))
    angle_deg = angle * 180/math.pi
    print('Angle = ', angle_deg)

    D1 = np.dot(V1, ray)
    d = D1 * math.tan(angle)
    print('d = ', d)

    a = math.sqrt(R**2 - d**2)
    print('a = ', a)

    b = D1 - a
    print('b = ', b)

    I = ray * b + C
    print('Intersection: ', I)
    return None

def intersection_ray_plane(P, N, C, ray):
    P = np.array(P)
    N = np.array(N)
    C = np.array(C)
    ray = np.array(ray)

    w = P - C
    print('w = ', w)
    a = np.dot(w, N)
    print('a = ', a)
    b = np.dot(ray, N)
    print('b = ', b)
    k = a/b
    print('k = ', k)
    I = ray * k + C
    print('Intersection: ', I)

    return I

def bezier_spline(P0, P1, P2):
    P0 = np.array(P0)
    P1 = np.array(P1)
    P2 = np.array(P2)

    R1 = ((P2 - P0)*1/6) + P1
    L2 = (P1 - R1) + P1
    L1 = ((P0 - P1) * (1.0/3.0)) + L2
    R2 = ((P2 - P1) * (1.0/3.0)) + R1
    print('L1: ', L1)
    print('L2: ', L2)
    print('R1: ', R1)
    print('R2: ', R2)
    return None

def barycentric_coordinate(a, b, c, p):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    p = np.array(p)

    A_t_cross = 1/2 * np.cross((b - a), (c - a))
    A_t = np.sqrt(np.sum(A_t_cross**2))

    A_a_cross = 1/2 * np.cross((c - p), (b - p))
    A_a = np.sqrt(np.sum(A_a_cross**2))
    A_b_cross = 1/2 * np.cross((a - p), (c - p))
    A_b = np.sqrt(np.sum(A_b_cross**2))
    A_c_cross = 1/2 * np.cross((a - p), (b - p))
    A_c = np.sqrt(np.sum(A_c_cross**2))

    return [A_a/A_t, A_b/A_t, A_c/A_t]

def intersection_edge(a, b, n, p):
    # useful when calculating point outside clipping plane
    a = np.array(a)
    b = np.array(b)
    p = np.array(p)

    d1 = np.dot(n, (a - p))
    d2 = np.dot(n, (b - p))
    t = d1/(d1 - d2)
    d = a + t * (b - a)
    return (d)

def refraction_angle(theta_1, n1, n2):
    angle = theta_1 * math.pi / 180
    angle_2 = math.asin(math.sin(angle) * n1 / n2)
    theta_2 = angle_2 * 180 / math.pi
    return theta_2

def rotate_vector(v1, angle):
    rad = angle * math.pi / 180
    rotation_mat = [
        [ math.cos(rad), math.sin(rad), 0],
        [-math.sin(rad), math.cos(rad), 0],
        [0,             0,              1],
    ]
    mat = np.asarray(rotation_mat)
    return (np.dot(mat, v1))

def interpolation (a, b, p, a_n, b_n):
    d1 = math.sqrt(np.sum((np.asarray(a) - np.asarray(p)) ** 2))
    d2 = math.sqrt(np.sum((np.asarray(b) - np.asarray(p)) ** 2))
    # if color:
    # return d2/(d1+d2) * np.asarray(n1) + d1/(d1+d2) * np.asarray(n2)
    # if normal:
    return (normalize(d2/(d1+d2) * np.asarray(a_n) + d1/(d1+d2) * np.asarray(b_n)))

def refraction_calculator(ray, camera, N1, N2, v1, v2, n1, n2, n):
    intersection = intersection_ray_plane(v1, n, camera, ray)
    print('refraction intersection: ', intersection)

    d1 = math.sqrt(np.sum((np.asarray(v1) - intersection) ** 2))
    d2 = math.sqrt(np.sum((np.asarray(v2) - intersection) ** 2))
    normal = normalize(d2/(d1+d2) * np.asarray(n1) + d1/(d1+d2) * np.asarray(n2))
    print("interpolated normal: ", normal)

    intersection_angle = math.pi - math.acos(np.dot(np.asarray(ray), normal))
    intersection_angle = intersection_angle * 180/math.pi
    print('intersection angle: ', intersection_angle)

    refraction_ang = refraction_angle(intersection_angle, N1, N2)
    print('refraction angle: ', refraction_ang)

    second_ray = rotate_vector(-normal, -refraction_ang)
    print('second ray direction: ', second_ray)


# ray = [-8.8073, 31.1927, 0]
# camera = [-31.7668, 27.1443, 0]
# N1 = 2.42
# N2 = 1.33
# v1 = [40, 0, 0]
# v2 = [0, -40, 0]
# n1 = [-0.707, 0.707, 0]
# n2 = [0, 1, 0]
# n = [-0.707, 0.707, 0]
# refraction_calculator(ray, camera, N1, N2, v1, v2, n1, n2, n)

def ray_box_intersection(min, max, C, r):
    min = np.array(min)
    max = np.array(max)
    r = np.array(r)
    C = np.array(C)
    R_prime = C + 1000 * r

    min_y_a = abs(min[1] - C[1])
    min_y_k = min_y_a / abs(R_prime[1] - C[1])
    min_y_I = (R_prime - C) * min_y_k + C

    min_x_a = abs(min[0] - C[0])
    min_x_k = min_x_a / abs(R_prime[0] - C[0])
    min_x_I = (R_prime - C) * min_x_k + C

    max_y_a = abs(max[1] - C[1])
    max_y_k = max_y_a / abs(R_prime[1] - C[1])
    max_y_I = (R_prime - C) * max_y_k + C

    max_x_a = abs(max[0] - C[0])
    max_x_k = max_x_a / abs(R_prime[0] - C[0])
    max_x_I = (R_prime - C) * max_x_k + C

    return min_x_I, min_y_I, max_x_I, max_y_I

# min = [0, 0]
# max = [20, 20]
# C = [-5.6,-7.1]
# r = [0.842403,0.538849]
# print(ray_box_intersection(min, max, C, r))

def distance_to_torus(C, r, torusN, torusC, torusR1, torusR2):
    C = np.array(C)
    r = np.array(r)
    torusN = np.array(torusN)
    torusC = np.array(torusC)
    torusR1 = np.array(torusR1)
    torusR2 = np.array(torusR2)

    k = np.dot((C - torusC), torusN)
    p = C - (torusN * k)
    # print(p)
    M = (p - torusC) / math.sqrt(np.sum((p - torusC) ** 2))
    m = M * torusR1 + torusC
    # print(m)
    D = math.sqrt(np.sum((m - C) ** 2)) - torusR2

    return D

torusC = [0, 0, 0]
torusN = [0, 0, 1]
torusR1 = 3
torusR2 = 0.5
C = [5, 5, 5]
r = [-1, -1, -1]
print(distance_to_torus(C, r, torusN, torusC, torusR1, torusR2))
# a = [1.7, -0.4, 0]
# b = [-0.8, 0.6, 0]
# n = [-1, 0 ,0]
# p = [1, 0, 0]
# =============== clipping ===================
# print(intersection_edge(a, b, n, p))

# a = [0.5, -0.8, 0.6]
# b = [0.55, 1, -0.02]
# c = [-0.7, -0.5, -0.5]
# p = [0, 0, -0.104]
# print(barycentric_coordinate(a,b,c,p))

# Sphere_center = [0,0,0]
# Radius = 35
# Camera = [12,-80,-5]
# ray = [0,1,0]
#intersection_ray_sphere(Sphere_center, Radius, Camera, ray)

# P = [90,-60,0]
# N = [-1, 0, 0]
# C = [27.1904, -12.8096, 0]
# ray = [0.8622, -0.5066, 0]
# # intersection_ray_plane(P, N, C, ray)

# v2 = [90, 30, 0]
# n1 = [0, 255, 0]
# n2 = [0, 0, 255]
# intersection = np.asarray([90, -49.7153, 0])
# d1 = math.sqrt(np.sum((np.asarray(P) - intersection) ** 2))
# d2 = math.sqrt(np.sum((np.asarray(v2) - intersection) ** 2))
# normal = d2/(d1+d2) * np.asarray(n1) + d1/(d1+d2) * np.asarray(n2)
# # print(normal)
#
# p0 = [15.415, 31.499, 0]
# p1 = [-8.3, 4.2, 0]
# p2 = [17, -1.9, 0]
# bezier_spline(p0, p1, p2)
