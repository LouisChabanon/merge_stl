import numpy as np
from stl import mesh
from shapely.geometry import Polygon, box
import matplotlib.pyplot as plt


def changment_de_base_ACP(input_path, output_path):
    """
    2e méthode de changement de base : ACP (Analyse de Composantes Principales)
    https://fr.wikipedia.org/wiki/Analyse_en_composantes_principales

    input_path: chemin vers le fichier STL d'entrée
    output_path: chemin vers le fichier STL de sortie

    retourne le fichier STL dans un nouveau repère

    problème: la direction de la pièce en sortie n'est pas forcement la même que dans le fichier d'entrée
    """

    m = mesh.Mesh.from_file(input_path)
    pts = m.vectors.reshape(-1, 3)

    centre_gravite = pts.mean(axis=0)
    centered_pts = pts - centre_gravite

    # Décomposition en valeurs propres (SVD)
    # https://fr.wikipedia.org/wiki/D%C3%A9composition_en_valeurs_singuli%C3%A8res
    _, _, vt = np.linalg.svd(centered_pts, full_matrices=False)
    # Direction principales: e1, plus grande dilatation, e3, plus petite dilatation (normale de la surface)
    e1, e2, e3 = vt

    # On s'assure que la base est directe
    if np.dot(np.cross(e1, e2), e3) < 0:
        e2 = -e2
    if e3[2] < 0:
        e3 = -e3

    # Matrice de rotation
    R = np.vstack([e1, e2, e3]).T

    pts_rot = centered_pts @ R

    # On place l'origine du repère au coin inférieur gauche de la boîte englobante
    min_x, min_y, min_z = pts_rot[:, 0].min(
    ), pts_rot[:, 1].min(), pts_rot[:, 2].min()
    origin = np.array([min_x, min_y, min_z])

    # 6) Translation pour placer l'origine à (0,0,0)
    pts_final = pts_rot - origin

    m.vectors = pts_final.reshape(-1, 3, 3)
    m.save(output_path)


def get_fill_per_pixel(mesh, voxel_size):

    pts = mesh.vectors.reshape(-1, 3)
    min_z = pts[:, 2].min()
    max_z = pts[:, 2].max()
    print(f"min_z: {min_z}, max_z: {max_z}")
    eps = 2
    mean_z_per_tri = mesh.vectors[:, :, 2].mean(axis=1)
    bottom_mask = np.abs(mean_z_per_tri - min_z) < eps
    bottom_tri = mesh.vectors[bottom_mask]

    if bottom_tri.size == 0:
        raise ValueError("No bottom triangle found")
    else:
        print(f"Found {len(bottom_tri)} bottom triangles")

    tri_polys = [Polygon(tri[:, :2]) for tri in bottom_tri]
    min_x, min_y = pts[:, 0].min(), pts[:, 1].min()
    max_x, max_y = pts[:, 0].max(), pts[:, 1].max()

    width_x = max_x - min_x
    width_y = max_y - min_y

    num_x = int(np.ceil(width_x / voxel_size))
    num_y = int(np.ceil(width_y / voxel_size))

    cell_area = voxel_size**2

    fill_map = np.zeros((num_x, num_y), dtype=float)

    for i in range(num_x):
        x0 = min_x + i * voxel_size
        x1 = x0 + voxel_size
        for j in range(num_y):
            yo = min_y + j * voxel_size
            y1 = yo + voxel_size
            cell = box(x0, yo, x1, y1)
            intersect = sum(cell.intersection(poly).area for poly in tri_polys)
            fill_map[i, j] = min(intersect / cell_area, 1.0)

        return fill_map


def visualize_fill_map(fill_map):
    """
    Display the fill map as a heatmap using matplotlib.

    Parameters
    ----------
    fill_map : np.ndarray
        The 2D array of fill percentages to visualize.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(fill_map.T, origin='lower',
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Fill Percentage')
    plt.title('2D Fill Map of Bottom Surface')
    plt.xlabel('X Pixels')
    plt.ylabel('Y Pixels')
    plt.tight_layout()
    plt.show()


def main():
    """
    Fonction principale
    """
    # for filename in os.listdir("sliced_stl"):
    #    if filename.endswith(".stl"):
    #        input_path = os.path.join("sliced_stl", filename)
    #        output_path = os.path.join(
    #            "rebased_stl", filename.replace(".stl", "_transformed_PCA.stl"))
    #        changment_de_base_ACP(input_path, output_path)
    #        print(f"Transformed {filename} to {output_path}")

    input_path = "rebased_stl/P1_transformed_PCA.stl"
    m = mesh.Mesh.from_file(input_path)
    fill_map = get_fill_per_pixel(m, 2)
    visualize_fill_map(fill_map)


if __name__ == "__main__":
    main()
