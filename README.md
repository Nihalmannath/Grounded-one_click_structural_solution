# Grounded structural solutions
A Structural Prediction Tool for Early-Stage Architectural Design

![Screenshot 2025-07-06 125029](https://github.com/user-attachments/assets/47e4fe0a-593f-479a-8d31-cb669f6cbc92)

Overview:
Grounded Structural Solutions is a predictive design tool built to assist architects during the conceptual design phase. It offers early structural feedback by processing simplified 3D models and suggesting optimal structural strategies—without compromising creativity.

How It Works:
Users upload a .glb or .obj file consisting of orthogonal volumes (e.g., boxes, extrusions, aligned forms). The tool processes this geometry through several automated stages:

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
   
![Screenshot 2025-07-06 124005](https://github.com/user-attachments/assets/30f0e2a6-820d-44de-858f-73b515cb5b02)

![Screenshot 2025-07-06 124028](https://github.com/user-attachments/assets/b90521f5-b0fc-48d7-abb3-d75732b03d25)


Output:
JSON/CSV of detected structural elements (walls, floors, columns)
Structural grid visualized

![Screenshot 2025-07-06 141224](https://github.com/user-attachments/assets/3b896688-e541-4857-b79b-5688208dc4fb)

PyNite-based summary report: displacements, reaction forces, and warnings

![Screenshot 2025-07-06 141525](https://github.com/user-attachments/assets/7c22862c-0ae1-448f-aa65-cf5021bdb2e5)


Developed as part of the Master in AI for Architecture & the Built Environment 2024–25, during the 2nd and 3rd term research studio.
Institution: IAAC – Institute for Advanced Architecture of Catalonia
Special Thanks: Faculty and advisors at IAAC for technical and theoretical guidance.
Refer for more info: https://blog.iaac.net/grounded-structural-generation-tool-for-architects/



