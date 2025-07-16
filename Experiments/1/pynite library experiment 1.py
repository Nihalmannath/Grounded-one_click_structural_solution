"""
Example of a simply supported beam with a uniform distributed load.
Units used in this example are inches and kips.
This example does not use load combinations. The program will create a
default load combination called 'Combo 1'.
"""

# Import `FEModel3D` from `Pynite`
from Pynite import FEModel3D

# Create a new finite element model
beam = FEModel3D()

# Add nodes (14 ft = 168 inches apart)
beam.add_node('N1', 0, 0, 0)
beam.add_node('N2', 168, 0, 0)

# Define a material
E = 29000       # Modulus of elasticity (ksi)
G = 11200       # Shear modulus of elasticity (ksi)
NU = 0.3        # Poisson's ratio
RHO = 2.836e-4  # Density (kci)
beam.add_material('Steel', E, G, NU, RHO)

# Add a section with the following properties:
# Iy = 100 in^4, Iz = 150 in^4, J = 250 in^4, A = 20 in^2
beam.add_section('MySection', 20, 100, 150, 250)

# Add a member
beam.add_member('M1', 'N1', 'N2', 'Steel', 'MySection')

# Provide simple supports
beam.def_support('N1', True, True, True, False, False, False)
beam.def_support('N2', True, True, True, True, False, False)

# Add a uniform load of 200 lbs/ft to the beam (from 0 in to 168 in)
beam.add_member_dist_load('M1', 'Fy', -200/1000/12, -200/1000/12, 0, 168)

# Alternatively the following line would do apply the load to the full
# length of the member as well
# beam.add_member_dist_load('M1', 'Fy', 200/1000/12, 200/1000/12)

# Analyze the beam
beam.analyze()

# Print the shear, moment, and deflection diagrams
beam.members['M1'].plot_shear('Fy')
beam.members['M1'].plot_moment('Mz')
beam.members['M1'].plot_deflection('dy')

# Print reactions at each end of the beam
print('Left Support Reaction:', beam.nodes['N1'].RxnFY, 'kip')
print('Right Support Reacton:', beam.nodes['N2'].RxnFY, 'kip')

# Render the deformed shape of the beam magnified 100 times, with a text
# height of 5 inches
renderer = Renderer(beam)
renderer.annotation_size = 6
renderer.deformed_shape = True
renderer.deformed_scale = 100
renderer.render_loads = True
renderer.render_model()

# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D

# # Create a new finite element model (like assembling LEGO blocks)
# beam = FEModel3D()

# # Define nodes (points in space, like LEGO connectors)
# beam.add_node("N1", 0, 0, 0)      # Left support (fixed)
# beam.add_node("N2", 168, 0, 0)    # Right support

# # Define a material (like the LEGO plastic strength)
# E = 29000       # Modulus of elasticity (ksi)
# G = 11200       # Shear modulus of elasticity (ksi)
# NU = 0.3        # Poisson's ratio
# RHO = 2.836e-4  # Density (kci)
# beam.add_material("Steel", E, G, NU, RHO)

# # Define the beam section (cross-section like LEGO beam size)
# beam.add_section("MySection", 20, 100, 150, 250)

# # Add a member (beam between two LEGO nodes)
# beam.add_member("M1", "N1", "N2", "Steel", "MySection")

# # Provide simple supports (fixing the LEGO to a base)
# beam.def_support("N1", True, True, True, False, False, False)  # Pinned
# beam.def_support("N2", True, True, True, True, False, False)   # Roller

# # Apply a uniform load of 200 lbs/ft (converted to kips per inch)
# beam.add_member_dist_load("M1", "Fy", -200/1000/12, -200/1000/12, 0, 168)

# # Run structural analysis (calculating deflections)
# beam.analyze()

# # Get results for visualization
# N1_x, N1_y, N1_z = 0, 0, 0
# # Note: Multiplying by 500 is for visualization purposes only to exaggerate the deflection
# N2_y = beam.nodes["N2"].dy * 500
# N2_x, N2_z = 168, 0  # Scaled for visibility

# # Visualization using Matplotlib 3D (LEGO-style)
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection="3d")

# # Original Beam (black line)
# ax.plot([N1_x, 168], [N1_y, 0], [N1_z, 0], color="black", linewidth=3, label="Original Beam")

# # Deformed Beam (dashed red line)
# ax.plot(
# 	[N1_x, N2_x],
# 	[N1_y, N2_y],
# 	[N1_z, N2_z],
# 	color="red",
# 	linewidth=3,
# 	linestyle="dashed",
# 	label="Deformed Beam",
# )

# # Nodes as LEGO-like blocks
# ax.scatter(
# 	[N1_x, N2_x],
# 	[N1_y, N2_y],
# 	[N1_z, N2_z],
# 	color="blue",
# 	s=200,
# 	marker="s",
# 	label="LEGO Nodes",
# )

# ax.quiver(
# 	N2_x,
# 	N2_y,
# 	N2_z,
# 	0,
# 	-1,
# 	0,
# 	color="green",
# 	length=10 / 12,
# 	normalize=True,
# 	label="Applied Load",
# )
# ax.quiver(
# 	N2_x,
# 	N2_y,
# 	N2_z,
# 	0,
# 	-1,
# 	0,
# 	color="green",
# 	length=10,
# 	normalize=True,
# 	label="Applied Load",
# )

# # Labels & Aesthetics

# ax.set_xlabel("X Axis (Length)")
# ax.set_ylabel("Y Axis (Deflection)")
# ax.set_zlabel("Z Axis")
# ax.set_title("LEGO-Style Structural Analysis Visualization")
# ax.legend()
# plt.show()

