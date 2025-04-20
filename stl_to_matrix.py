import numpy as np
from stl import mesh
import os


"""
def find_refs_points(stl):
    \"""
    Trouve les points de références pour la transformation:
    \"""
    pts = stl.vectors.reshape(-1, 3)
    y_min = pts[:, 1].min()

    # Pas sûr que ce soit la bonne méthode pour trouver le plan de base
    eps = (pts[:, 1].max() - y_min) * 1e-3
    base_plane = pts[np.abs(pts[:,2] - y_min) < eps]
    # On prend le point le plus à gauche et le plus à droite de la base
    x_min, x_max = base_plane[:, 0].min(), base_plane[:, 0].max()
    z_min, z_max = base_plane[:, 2].min(), base_plane[:, 2].max()

    origin = np.array([x_min, y_min, z_min])
    x_ref = np.array([x_max, y_min, z_min])
    z_ref = np.array([x_min, y_min, z_max])

    return origin, x_ref, z_ref


def changement_de_base(origin, x_ref, y_ref):
    \"""
    depuis 3 points de la base d'origine :
     origin: point (0,0,0) dans le nouveau repère
     x_ref: point dans la direction X
     y_ref: point dans la direction Y

    retourne la translation et la matrice de rotation
    \"""


    translation = np.array(origin)

    v_x = np.array(x_ref) - translation
    v_y = np.array(y_ref) - translation

    e_x = v_x / np.linalg.norm(v_x)
    e_y = v_y / np.linalg.norm(v_y)

    e_z = np.cross(e_x, e_y)

    R = np.array([e_x, e_y, e_z]).T
    return translation, R
"""

def changment_de_base_ACP(input_path, output_path):
    """
    2e méthode de changement de base : ACP (Analyse de Composantes Principales)
    https://fr.wikipedia.org/wiki/Analyse_en_composantes_principales
    """

    m = mesh.Mesh.from_file(input_path)
    pts = m.vectors.reshape(-1, 3)

    centre_gravite = pts.mean(axis=0)
    centered_pts = pts - centre_gravite

    # Décomposition en valeurs propres (SVD)
    #https://fr.wikipedia.org/wiki/D%C3%A9composition_en_valeurs_singuli%C3%A8res
    _, _, vt = np.linalg.svd(centered_pts, full_matrices=False)
    # Direction principales
    e1, e2, e3 = vt

    # On s'assure que la base est directe
    if np.dot(np.cross(e1, e2), e3) < 0:
        e2 = -e2
    if e3[2] < 0:
        e3 = -e3

    # Matrice de rotation
    R = np.vstack([e1, e2, e3]).T

    pts_rot = centered_pts @ R

    z_min = pts_rot[:, 2].min()
    eps = (pts_rot[:, 2].max() - z_min) * 1e-3
    base = pts_rot[np.abs(pts_rot[:, 2] - z_min) < eps]

    min_x, max_x = base[:, 0].min(), base[:, 0].max()
    min_y, max_y = base[:, 1].min(), base[:, 1].max()
    origin = np.array([min_x, min_y, z_min])

    # 6) Translation pour placer l'origine à (0,0,0)
    pts_final = pts_rot - origin

    m.vectors = pts_final.reshape(-1, 3, 3)
    m.save(output_path)
    



def main():
    """
    Fonction principale
    """
    for filename in os.listdir("sliced_stl"):
        if filename.endswith(".stl"):
            input_path = os.path.join("sliced_stl", filename)
            output_path = os.path.join("rebased_stl", filename.replace(".stl", "_transformed_PCA.stl"))
            changment_de_base_ACP(input_path, output_path)
            print(f"Transformed {filename} to {output_path}")


if __name__ == "__main__":
    main()
