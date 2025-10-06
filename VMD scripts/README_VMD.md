# DynMoCo VMD Visualization Guide

This guide explains how to visualize dynamic community assignments from DynMoCo analysis using VMD (Visual Molecular Dynamics). The visualization tools allow you to observe structural rearrangements and community dynamics over time in molecular trajectories.

## Overview

We provide Python scripts that generate TCL scripts for VMD to visualize dynamic community assignments of residues over time. The visualization features:

- **Community coloring**: Each community is assigned a unique color from a palette of 17 distinct colors
- **Community annotations**: Community IDs (C1, C2, etc.) are displayed near corresponding residues
- **Frame synchronization**: Community assignments update automatically as you navigate through the trajectory
- **Interactive selection**: Use VMD commands to select, hide, or highlight specific communities

## Prerequisites

1. **VMD Installation**: Download and install VMD from [https://www.ks.uiuc.edu/Research/vmd/](https://www.ks.uiuc.edu/Research/vmd/)
2. **Required Files**: Ensure you have the following files:
   - Community assignments JSON file (from `communities/` folder)
   - Structure file (.pdb or .gro file from `data/` folder)
   - Trajectory file (.xtc file from `data/` folder)
   - Python script (`VMD scripts/json_to_tcl.py`)


## Quick Start

### Step 1: Generate TCL Script

1. Navigate to the `VMD scripts/` directory
2. Edit `json_to_tcl.py` and update the parameters:

```python
# User Parameters Section
json_file = "communities/a5b1_clamp_communities.json"  # Path to your JSON file
output_tcl = "a5b1_clamp_example.tcl"                  # Output TCL filename
selection_type = "residue"                             # Selection type
ntimestamps = 4                                        # Number of timesteps
```

3. Run the script:
```bash
python json_to_tcl.py
```

This generates a TCL file (e.g., `a5b1_clamp_example.tcl`) ready for VMD.

### Step 2: Load in VMD

1. **Load Structure File**:
   - File → New Molecule → Browse to your `.gro` or `.pdb` file
   - Click "Load"

2. **Load Trajectory**:
   - File → New Molecule → Browse to your `.xtc` file
   - Click "Load"

3. **Load TCL Script**:
   - Extensions → TK Console
   - Type: `source /full/path/to/your/tcl/file.tcl`
   - Press Enter

## JSON File Format

The community assignment JSON file should follow this structure:

```json
{
  "0": {  # timesteps
    "C1": [1, 2, 3, 4], # community ID: [resindices]
    "C2": [5, 6, 7, 8],
    "C3": [9, 10, 11]
  },
  "1": {
    "C1": [1, 2, 3, 5],
    "C2": [4, 6, 7, 8],
    "C3": [9, 10, 11]
  }
}
```

## Visualization Features

### Dynamic Community Coloring

Communities are automatically colored using a predefined palette of 17 distinct colors. Colors update as you navigate through the trajectory frames.

### Community Selection

The TCL script assigns community numbers to VMD user variables (`user`, `user2`, `user3`, `user4`) for each timestep. Use these for interactive selection:

```tcl
# Select community C5 at timestep 4
set my_selection [atomselect top "user4 == 5"]

# Hide community C3 at timestep 2
set hide_sel [atomselect top "user2 == 3"]
$hide_sel set occupancy 0

# Show only communities C1 and C2 at timestep 1
set show_sel [atomselect top "user2 == 1 or user2 == 2"]
```

### Community Annotations

Community IDs are displayed as text labels positioned near the corresponding residues, making it easy to identify and track communities throughout the trajectory.

## Example Visualizations

### Community Overview
The visualization shows all communities with distinct colors, allowing you to observe global conformational changes and community dynamics.

![Community Overview](https://github.com/user-attachments/assets/0db4b392-546e-4738-8e73-6443049e903d)

### Specific Community Interactions
Focus on specific community interactions by highlighting communities of interest while showing others in neutral colors.

![AlphaIIb Beta3 C1-C2 Interaction](https://github.com/user-attachments/assets/757abe96-1e6e-4978-8d98-7e9ef6eba317)

### Pre-configured Visualization States

Load example visualization states for quick setup:

1. Load your structure and trajectory files
2. Load the visualization state: `alphaIIb_beta3_C1-C2 interaction.vmd`

This pre-configured state highlights specific community interactions with optimized viewing parameters.

## Customization

### Color Scheme
Modify colors in the TCL script by changing the color definitions:

```tcl
# Example color customization
set color1 "red"
set color2 "blue"
set color3 "green"
```

### Representation Styles
Adjust molecular representation styles:

```tcl
mol modstyle 0 top NewCartoon
mol modstyle 1 top Licorice
```

### Selection Criteria
Modify residue selection logic for different analysis needs:

```python
# In json_to_tcl.py
selection_type = "residue"  # or "atom", "backbone", etc.
```

## Troubleshooting

### Common Issues

1. **TCL script not loading**:
   - Check file path is correct and accessible
   - Ensure VMD TK Console is open
   - Verify TCL syntax in generated file

2. **Communities not displaying**:
   - Verify JSON file format matches expected structure
   - Check that residue numbers in JSON match structure file
   - Ensure trajectory and structure files are properly loaded

3. **Colors not updating**:
   - Navigate through trajectory frames to trigger updates
   - Check that community assignments exist for current timestep

### File Path Issues

- Use absolute paths when sourcing TCL files
- Ensure all referenced files (JSON, structure, trajectory) are accessible
- Check file permissions for read access

## Advanced Usage

### Batch Processing

Process multiple datasets by modifying the Python script to iterate over different JSON files:

```python
import glob

json_files = glob.glob("communities/*_communities.json")
for json_file in json_files:
    # Process each file
    output_tcl = json_file.replace('.json', '_viz.tcl')
    # ... rest of processing
```

### Custom Analysis

Combine VMD visualization with custom analysis scripts:

```tcl
# Example: Calculate community center of mass
set com_sel [atomselect top "user2 == 1"]
set com [measure center $com_sel]
puts "Community C1 center of mass: $com"
```

## Data Sources

All example data, JSON files, structure files, trajectories, and scripts for the models (α5β1, αIIbβ3, αVβ3, αVβ3) are available in the project repository and associated data folders.

For additional datasets and examples, refer to the main DynMoCo repository documentation.
