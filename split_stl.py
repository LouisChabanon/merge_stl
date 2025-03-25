import os 
import numpy as np
from stl import mesh, Mode


def detect_voxel_height(model):
    """
    Detecte la hauteur du voxel en analysant les différences entre les valeurs Z des triangles
    """
    z_values = np.unique(model.vectors[:, :, 2].flatten())
    z_differences = np.diff(np.sort(z_values))  # Differences entre les valeurs Z triées
    voxel_height = np.min(z_differences[z_differences > 0])  # Plus petite différence positive
    print(f"Detected voxel height: {voxel_height}")
    return voxel_height


def create_top_bottom_faces(inbound_faces, voxel_height, minx, maxx, miny, maxy, z_bottom, z_top):
    """
    Crée des faces top et bottom pour combler les zones "intérieures" manquantes dans la tranche.
    """
    pass

def main():
    """
    Découpe un fichier STL voxelisé en tranches horizontales et sauvegarde chaque tranche dans un fichier STL séparé
    """

    input_file = 'input.stl'
    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = mesh.Mesh.from_file(input_file)

    # Limites de la boîte englobante
    minx, maxx, miny, maxy, minz, maxz = model.x.min(), model.x.max(), model.y.min(), model.y.max(), model.z.min(), model.z.max()
    print(f"Limites de la boîte englobante: \n X: {minx} à {maxx}, \n Y: {miny} à {maxy}, \n Z: {minz} à {maxz}")

    slice_index = 0
    current_z = minz

    # Tranches de deux voxels de hauteur
    slice_size = 2*detect_voxel_height(model)

    while current_z < maxz:
        z_bottom = current_z
        z_top = current_z + slice_size

        # Sélectionne les triangles qui sont dans la tranche
        inbound_faces = model.vectors[
            (model.vectors[:, :, 2].min(axis=1) >= z_bottom) & (model.vectors[:, :, 2].max(axis=1) <= z_top)]

        slice_mesh = mesh.Mesh(np.zeros(inbound_faces.shape[0], dtype=mesh.Mesh.dtype))
        additional_faces = create_top_bottom_faces(inbound_faces, slice_size, minx, maxx, miny, maxy, z_bottom, z_top)
        

        slice_mesh.vectors = inbound_faces


        
        out_file = os.path.join(output_folder, f'slice_{slice_index}.stl')
        slice_mesh.save(out_file, mode=Mode.ASCII)
        print(f"Saved slice {slice_index} (X: {z_bottom} to {z_top}) with 4 triangles to {out_file}")

        slice_index += 1
        current_z += slice_size

if __name__ == '__main__':
    main()