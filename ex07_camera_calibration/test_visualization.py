import open3d as o3d

mesh_frames = []
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
mesh_frames.append(mesh_frame)
o3d.visualization.draw_geometries(mesh_frames)
