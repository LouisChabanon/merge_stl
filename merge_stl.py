import numpy as np
from stl import mesh 
import os 

def main():
    # Create a new empty mesh
    mesh1 = mesh.Mesh(np.zeros(0, dtype=mesh.Mesh.dtype))

    # Load the meshes
    for filename in os.listdir('fichier/.'):
        if filename.endswith('.stl'):
            mesh2 = mesh.Mesh.from_file(filename)
            

    
    # Write the mesh to file "combined.stl"
    mesh1.save('combined.stl')