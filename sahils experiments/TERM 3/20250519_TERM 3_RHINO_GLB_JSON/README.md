# Structural Modeling and Analysis Tool

## Overview
This project provides tools for 3D structural grid generation and analysis, supporting interactive visualization and adjustment of structural elements.

## Features
- Conversion of Rhino 3DM models to OBJ meshes
- Combination of multiple OBJ mesh layers
- Generation of 3D structural grids with:
  - Vertical columns
  - Horizontal beams
  - Interactive grid movement
  - JSON export of structural elements

## File Structure
```
.
├── docs/                 # Documentation and reference files
├── input model/          # Input model files (3DM, OBJ)
│   └── meshed_layers_obj/  # Individual mesh layers
├── output/               # Export directory for generated files
├── src/                  # Source code
│   ├── integrated_workflow.py  # Main integration script
│   ├── obj_combiner.py         # Mesh combining utilities
│   ├── Rhino_to_mesh.py        # 3DM to OBJ conversion
│   └── structural_grid.py      # Structural grid generation
└── requirements.txt      # Python dependencies
```

## Installation
1. Ensure Python 3.8+ is installed
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
Run the integrated workflow:
```
python src/integrated_workflow.py
```

For just the structural grid:
```
python src/structural_grid.py
```

## Input Formats
The tool supports the following file formats:
- Rhino 3DM files (.3dm)
- OBJ files (.obj)
- STL files (.stl)
- PLY files (.ply)
- GLB files (.glb)

## Output
- Structural elements are exported in JSON format
- 3D visualizations can be interactively adjusted

## Created
May 2025
