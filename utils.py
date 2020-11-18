import numpy as np
import math

def normalize(v):
    normalized_v = v / np.sqrt(np.sum(v**2))
    return normalized_v

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

    return None

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
    a = np.array(a)
    b = np.array(b)
    p = np.array(p)

    d1 = np.dot(n, (a - p))
    d2 = np.dot(n, (b - p))
    t = d1/(d1 - d2)
    d = a + t * (b - a)
    return (d)

a = [-0.7, -0.5, -0.5]
b = [0.6, 3, -0.7]
n = [0, -1 ,0]
p = [0, 1, 0]
# print(intersection_edge(a, b, n, p))

# a = [0.5, -0.8, 0.6]
# b = [0.55, 1, -0.02]
# c = [-0.7, -0.5, -0.5]
# p = [0, 0, -0.104]
# print(barycentric_coordinate(a,b,c,p))

Sphere_center = [5,-3,21]
Radius = 2
Camera = [79,-3,22]
ray = [-1,0,0]
# intersection_ray_sphere(Sphere_center, Radius, Camera, ray)

P = [4.74308,16.4974,31.8967]
N = [-0.0658428, 0.355551, 0.932335]
C = [19.7235,13.4488,74.5869]
ray = [0.408218,-0.258819,-0.875426]
# intersection_ray_plane(P, N, C, ray)

p0 = [-18.5, -1.7, 0]
p1 = [-3.5, 30, 0]
p2 = [19, 0.7, 0]
# bezier_spline(p0, p1, p2)
