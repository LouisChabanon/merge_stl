import numpy as np
from stl import mesh
import os


def open_voxels(input_folder) -> np.ndarray:
    """
    Load all STL files from the input folder and return them as a numpy array of meshes.
    """
    meshes = np.array([])
    for file in os.listdir(input_folder):
        if file.endswith('.stl'):
            meshes = np.append(meshes, mesh.Mesh.from_file(
                os.path.join(input_folder, file)))

    return meshes


def detect_voxels(voxel_size, meshes, matrix):
    """
    Detect the voxels in the meshes and fill the matrix accordingly.
    """
    for mesh in meshes:
        for triangle in mesh.vectors:
            tri_min = triangle.min(axis=0)
            tri_max = triangle.max(axis=0)

            idx_min = np.floor(tri_min / voxel_size).astype(int)
            idx_max = np.ceil(tri_max / voxel_size).astype(int)

    return matrix


def main():
    """
    Main function to load the STL files, detect voxels, and fill the matrix.
    """

    voxel_size = 2

    grid_shape = (200, 800, 200)
    matrix = np.zeros(grid_shape)
    input = "input.stl"
    mes = mesh.Mesh.from_file(input)
    # meshes = open_voxels(input)
    meshes = np.array([mes])
    filled_matrix = detect_voxels(voxel_size, meshes, matrix)
    print(filled_matrix)


if __name__ == "__main__":
    main()
