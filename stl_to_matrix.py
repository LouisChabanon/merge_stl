import numpy as np
from stl import mesh
import os


def find_mins_maxs(obj):
    minx = obj.x.min()
    maxx = obj.x.max()
    miny = obj.y.min()
    maxy = obj.y.max()
    minz = obj.z.min()
    maxz = obj.z.max()
    return minx, maxx, miny, maxy, minz, maxz


def get_bottom_corner(vertices):
    # Étape 1 : trouver la valeur minimale de X
    min_x = np.min(vertices[:, 0])
    candidates_x = vertices[vertices[:, 0] == min_x]

    # Étape 2 : parmi eux, trouver ceux avec la valeur minimale de Y
    min_y = np.min(candidates_x[:, 1])
    candidates_xy = candidates_x[candidates_x[:, 1] == min_y]

    # Étape 3 : parmi ceux-là, prendre celui avec le Z le plus élevé
    idx_max_z = np.argmax(candidates_xy[:, 2])
    result_point = candidates_xy[idx_max_z]

    return result_point


def changement_de_base(obj, z1, x1):
    """
    Change the base of the STL object to the new coordinate system defined by z1 and x1.
    """

    # Find the bottom corner
    bottom_corner = get_bottom_corner(obj.vectors)

    # Translate the STL object
    obj.vectors -= bottom_corner

    # Create a rotation matrix
    z_axis = z1 / np.linalg.norm(z1)
    x_axis = x1 / np.linalg.norm(x1)
    y_axis = np.cross(z1, x1)

    rotation_matrix = np.array([x_axis, y_axis, z_axis])

    # Rotate the STL object
    obj.rotate_using_matrix(rotation_matrix)


def main():
    """
    Main function to load the STL files, detect voxels, and fill the matrix.
    """

    stl = mesh.Mesh.from_file('sliced_stl/P1.stl')

    minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(stl)
    ref_point = np.array([minx, miny, minz])

    z1 = np.array([minx, maxy, maxz]) - ref_point
    x1 = np.array([maxx, miny, minz]) - ref_point

    print(x1, z1, ref_point)

    changement_de_base(stl, z1, x1)

    stl.save('sliced_stl/P1_transformed.stl')


if __name__ == "__main__":
    main()
