# Structural Grid Generator Documentation

## Overview
The Structural Grid Generator is a tool for creating 3D structural grids from mesh files.
It allows for interactive adjustment and visualization of structural elements.

## Main Components

### 1. File Import
- Supports multiple mesh formats (OBJ, STL, PLY, GLB)
- Conversion from Rhino 3DM files

### 2. Grid Generation
- Automatic grid point generation within mesh bounds
- Interactive adjustment of grid positioning
- Support for multiple floors with custom spacing

### 3. Structure Generation
- Vertical columns between floors
- Horizontal beams connecting grid points
- Internal filtering to ensure elements stay within the mesh bounds

### 4. Visualization
- 3D interactive visualization using PyVista
- Adjustable grid parameters via sliders
- Real-time feedback on grid adjustments

### 5. Export
- Export of structural elements to JSON format
- Compatible with external structural analysis tools

## JSON Export Format

```json
{
    "elements": {
        "beams": [
            {
                "start": [x, y, z],
                "end": [x, y, z]
            },
            ...
        ]
    }
}
```

## Workflow Integration
This tool integrates with:
1. Rhino 3DM to OBJ conversion
2. OBJ mesh combination
3. Structural grid generation

## Requirements
- Python 3.8+
- Dependencies in requirements.txt
