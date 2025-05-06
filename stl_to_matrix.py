import numpy as np
from stl import mesh
from shapely.geometry import Polygon, box
from shapely import STRtree
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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

    # 7) On nudge manuellement la pièce pour qu'elle soit dans le repère
    not_finished = True
    while not_finished:
        fig, ax = visualize_stl(m)
        inp = input("Select axis to flip (x, y, z) or 'done' to finish: ")
        plt.close(fig)
        if inp == "done":
            not_finished = False
        else:
            try:
                flip = [inp]
                m = nudge_part(m, translation=(0, 0, 0), flip=flip)
            except ValueError as e:
                print(f"Invalid input: {e}")
                continue
    # On place l'origine du repère au coin inférieur gauche de la boîte englobante
    min_x, min_y, min_z = pts_rot[:, 0].min(
    ), pts_rot[:, 1].min(), pts_rot[:, 2].min()
    origin = np.array([min_x, min_y, min_z])

    # 6) Translation pour placer l'origine à (0,0,0)
    pts_final = pts_rot - origin

    m.vectors = pts_final.reshape(-1, 3, 3)
    m.save(output_path)


def nudge_part(m, translation=(0, 0, 0), flip=None):
    """
    Translate et/ou retourne la pièce
    m : stl.mesh.Mesh Le maillage STL à analyser.
    translation : tuple La translation à appliquer (x, y, z).
    flip : tuple La direction de retournement à appliquer (x, y, z).
    """
    # Translation
    dx, dy, dz = translation
    m.vectors += np.array([dx, dy, dz])

    # Retourner la pièce
    if flip:
        for axis in flip:
            if axis == 'x':
                m.vectors[:, :, 0] *= -1
            elif axis == 'y':
                m.vectors[:, :, 1] *= -1
            elif axis == 'z':
                m.vectors[:, :, 2] *= -1
            else:
                raise ValueError(
                    "Invalid axis for flipping. Use 'x', 'y', or 'z'.")

    pts = m.vectors.reshape(-1, 3)
    min_coords = pts.min(axis=0)
    if np.any(min_coords < 0):
        # Si la pièce est en dehors du repère, on la translate pour qu'elle soit dans le repère
        m.vectors += np.abs(min_coords)
    return m


def get_fill_per_pixel(mesh, voxel_size, z=0, eps=0.5):
    """
    Calcule le pourcentage de remplissage pour chaque pixel d'une surface de normale z.

    mesh : stl.mesh.Mesh Le maillage STL à analyser.
    voxel_size : float La taille de chaque pixel (voxel) dans l'espace 2D.
    z : float La coordonnée z de la surface à analyser (par défaut 0).
    eps: float La tolérance pour la détection de la surface (par défaut 0.5). Pas sur de la valeur.

    retourne un tableau 2D représentant le pourcentage de remplissage pour chaque pixel.
    """

    pts = mesh.vectors.reshape(-1, 3)
    min_z = pts[:, 2].min()
    mean_z_per_tri = mesh.vectors[:, :, 2].mean(axis=1)

    # On ne garde que les triangles dont la coordonnée z est proche de la coordonnée z de la surface
    bottom_mask = np.abs(mean_z_per_tri - (min_z + z)) < eps
    bottom_tri = mesh.vectors[bottom_mask]

    if bottom_tri.size == 0:
        raise ValueError("No bottom triangle found")
    else:
        print(f"Found {len(bottom_tri)} bottom triangles")

    # On crée une liste de polygones à partir des triangles
    # On ne garde que les 2 premières dimensions (x,y) pour le calcul de l'intersection
    tri_polys = [Polygon(tri[:, :2]) for tri in bottom_tri]
    min_x, min_y = pts[:, 0].min(), pts[:, 1].min()
    max_x, max_y = pts[:, 0].max(), pts[:, 1].max()

    width_x = max_x - min_x
    width_y = max_y - min_y

    # On crée une grille de pixels (voxel) sur la surface
    num_x = int(np.ceil(width_x / voxel_size))
    num_y = int(np.ceil(width_y / voxel_size))

    cell_area = voxel_size**2

    fill_map = np.zeros((num_x, num_y), dtype=float)

    # Calcul de l'intersection entre chaque triangle et chaque cellule de la grille
    # Très, très lent (80x20pixels = 1600 cellules pour environ 28000 triangles = 44 millions d'intersections)
    # On peut faire mieux en utilisant un algorithme de recherche spatiale (ex: R-tree)
    # https://shapely.readthedocs.io/en/stable/strtree.html

    # On crée un arbre de recherche spatiale pour les polygones
    tree = STRtree(tri_polys)

    for i in range(num_x):
        x0 = min_x + i * voxel_size
        x1 = x0 + voxel_size
        for j in range(num_y):
            # print(f"Traitement de ({i}, {j}) sur ({num_x}, {num_y})")
            yo = min_y + j * voxel_size
            y1 = yo + voxel_size
            # Crée un polygone rectangle pour la cellule
            cell = box(x0, yo, x1, y1)

            # Trouve les polygones qui intersectent la cellule
            candidate_index = tree.query(cell)
            candidate = tree.geometries.take(
                candidate_index)  # Récupère les polygones

            # Calcule l'aire d'intersection entre la cellule et les polygones
            intersect = sum(cell.intersection(poly).area for poly in candidate)
            fill_map[i, j] = min(intersect / cell_area, 1.0)

    return fill_map


def visualize_stl(mesh):
    # Create a new plot
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')

    # Load the STL files and add the vectors to the plot
    axes.add_collection3d(Poly3DCollection(mesh.vectors))

    # Remove transparency
    for poly in axes.collections:
        poly.set_alpha(1.0)
        poly.set_edgecolor('k')
        poly.set_linewidth(0.1)
    # Auto scale to the mesh size
    scale = mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    # Show the plot to the screen
    figure.show()

    return figure, axes


def visualize_fill_map(fill_map, file_name, show=True):
    """
    Display the fill map as a heatmap using matplotlib.

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
    if show:
        plt.show()
    plt.savefig(file_name)


def compare_fill_maps(fill_maps_1, fill_maps_2):
    """
    Compare deux fill maps
    """
    if len(fill_maps_1) != len(fill_maps_2):
        print(
            f"Les deux fill maps n'ont pas le même nombre de plans: {len(fill_maps_1)} != {len(fill_maps_2)}")
        return

    for idx, (fm1, fm2) in enumerate(zip(fill_maps_1, fill_maps_2)):
        # Déterminer la taille cible
        max_x = max(fm1.shape[0], fm2.shape[0])
        max_y = max(fm1.shape[1], fm2.shape[1])

        # Créer des tableaux padded
        pad1 = np.zeros((max_x, max_y), dtype=float)
        pad2 = np.zeros((max_x, max_y), dtype=float)
        pad1[:fm1.shape[0], :fm1.shape[1]] = fm1
        pad2[:fm2.shape[0], :fm2.shape[1]] = fm2

        diff = np.abs(pad1 - pad2)
        print(f"Différence moyenne entre les fill maps {idx}: {diff.mean()}")

        # Visualisation
        visualize_fill_map(diff, f"fill_diff/diff_{idx}.png", show=False)


def main():
    """
    Fonction principale
    """
    files1 = sorted(os.listdir("rebased_stl"))
    files2 = sorted(os.listdir("rebased_stl_2"))
    """
    for filename in os.listdir("sliced_stl_2"):
        if filename.endswith(".stl"):
            input_path = os.path.join("sliced_stl_2", filename)
            output_path = os.path.join(
                "rebased_stl_2", filename.replace(".stl", "_transformed_PCA.stl"))
            changment_de_base_ACP(input_path, output_path)
            print(f"Transformed {filename} to {output_path}")
    """
    fill_maps_1 = []

    for filename in files1:
        if filename.endswith("_transformed_PCA.stl"):
            file_path = os.path.join("rebased_stl", filename)
            m = mesh.Mesh.from_file(file_path)
            voxel_size = 2
            plans = [0, voxel_size*2]
            for i in plans:
                fill_map = get_fill_per_pixel(m, voxel_size, z=i, eps=1)
                fill_maps_1.append(fill_map)
                visualize_fill_map(
                    fill_map, f"fill_map/{filename}_{i}.png", show=False)

    fill_maps_2 = []

    for filename in files2:
        if filename.endswith("_transformed_PCA.stl"):
            file_path = os.path.join("rebased_stl_2", filename)
            m = mesh.Mesh.from_file(file_path)
            voxel_size = 2
            plans = [0, voxel_size*2]
            for i in plans:
                fill_map = get_fill_per_pixel(m, voxel_size, z=i, eps=1)
                fill_maps_2.append(fill_map)
                visualize_fill_map(
                    fill_map, f"fill_map_2/{filename}_{i}.png", show=False)

    compare_fill_maps(fill_maps_1, fill_maps_2)


if __name__ == "__main__":
    main()
