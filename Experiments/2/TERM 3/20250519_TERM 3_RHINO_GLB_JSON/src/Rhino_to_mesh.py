# 1) Install dependency
import rhino3dm
import os

# 2) Load the 3dm
input_path = r'./input model/rectangles3dcantilever.3dm'
model = rhino3dm.File3dm.Read(input_path)

# 3) Helper to mesh a Brep
def mesh_brep(brep, mesh_type=rhino3dm.MeshType.Default):
    meshes = []
    for face in brep.Faces:
        try:
            m = face.GetMesh(mesh_type)
        except Exception:
            continue
        if m:
            meshes.append(m)
    return meshes

# 4) Prepare output directory
out_dir = r'./input model/meshed_layers_obj'
os.makedirs(out_dir, exist_ok=True)

# 5) Iterate through layers and export OBJ per layer
for lyr in model.Layers:
    meshes = []
    for obj in model.Objects:
        if obj.Attributes.LayerIndex != lyr.Index:
            continue
        geom = obj.Geometry
        if isinstance(geom, rhino3dm.Mesh):
            meshes.append(geom)
        elif isinstance(geom, rhino3dm.Brep):
            meshes.extend(mesh_brep(geom))

    print("Layer [{}] '{}' -> {} mesh(es)".format(lyr.Index, lyr.Name, len(meshes)))
    if not meshes:
        continue

    # Write OBJ file
    safe_name = lyr.Name.replace(' ', '_')
    obj_path = os.path.join(out_dir, f'meshed_layer_{lyr.Index}_{safe_name}.obj')
    with open(obj_path, 'w') as f:
        # write a comment header
        f.write(f"# Layer {lyr.Index}: {lyr.Name}\n")
        vertex_offset = 0

        # write vertices and faces
        for mesh in meshes:
            # vertices
            for v in mesh.Vertices:
                f.write("v {} {} {}\n".format(v.X, v.Y, v.Z))
            # faces
            for face in mesh.Faces:
                if isinstance(face, tuple):
                    idxs = face
                else:
                    if getattr(face, "IsQuad", False):
                        idxs = (face.A, face.B, face.C, face.D)
                    else:
                        idxs = (face.A, face.B, face.C)
                # OBJ indexing is 1-based
                idxs = [i + 1 + vertex_offset for i in idxs]
                f.write("f {}\n".format(" ".join(map(str, idxs))))
            vertex_offset += len(mesh.Vertices)

    print("  Saved OBJ:", obj_path)

print("Done. Check the folder './input model/meshed_layers_obj' for your OBJ files.")