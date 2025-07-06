# Grounded structural solutions
A Structural Prediction Tool for Early-Stage Architectural Design

Overview:
Grounded Structural Solutions is a predictive design tool built to assist architects during the conceptual design phase. It offers early structural feedback by processing simplified 3D models and suggesting optimal structural strategies—without compromising creativity.

How It Works:
Users upload a .obj file consisting of orthogonal volumes (e.g., boxes, extrusions, aligned forms). The tool processes this geometry through several automated stages:

Features:
1. Mesh segmentation with trimesh
  Automatically detects and separates walls, floors, and roofs from the input mesh4. Structural grid formation with custom algorithm
  Filters out non-load-bearing elements
2. Structural Grid Formation
   Generates a custom structural grid based on floorplate size and wall intersections
   Adapts column placement rules to geometry constraints (e.g., perimeter-first logic)
3. Structural Assessment with PyNite
   Applies finite element analysis (FEA) to estimate displacement and stress
   Outputs a report on load paths, floor stiffness, and structural weaknesses

Output:
JSON/CSV of detected structural elements (walls, floors, columns)
Structural grid visualized with matplotlib or Rhino3DM (optional)
PyNite-based summary report: displacements, reaction forces, and warnings


Developed as part of the Master in AI for Architecture & the Built Environment 2024–25, during the 2nd and 3rd term research studio.
Institution: IAAC – Institute for Advanced Architecture of Catalonia
Special Thanks: Faculty and advisors at IAAC for technical and theoretical guidance.



