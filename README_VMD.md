# Objective
This project provides a Python script that generates a Tcl script for VMD to visualize dynamic community assignments of residues over time. Given a JSON file containing residue assignments to communities across multiple timestamps, the script enables users to color-code residues according to their community membership for each frame of a trajectory.

Each community is annotated with a number in the VMD model. The number is positioned near the corresponding residues. Each community is assigned a unique color from a set of 17 distinct colors. This makes communities easy to distinguish and track throughout the trajectory. The visualization allows you to observe structural rearrangements over time. It also supports detailed analysis of local residue interactions and global conformational changes.

# Required Inputs
- JSON file: Contains community assignments for residues over time frames. (located in communities folder)
  Example format: {"0": {"C1": [0,1,2]}} 
- Structure file: .pdb or .gro file (located in data folder)
- Trajectory file: .xtc file (located in data folder)
- Python file: Generate tcl file from JSON file (located in scripts folder)

# How to Use Python to Generate TCL
- Make sure the JSON file is saved in the same folder as the Python script.
- Open the Python file (py_to_tcl.py) in VS Code, IntelliJ, or Notepad++.
- Update the User Parameters section in the Python file with the correct file paths and variables.
- Run the Python script. This will generate a TCL file. Example result: a2b3_example.tcl

# HOW TO LOAD TCL in VMD
1. Load the structure file (.pdb or .gro file) in VMD: File -> New Molecule -> [filename]
2. Load the trajectory file (.xtc) on top of the structure file: File -> New Molecule -> [filename]
3. Source the TCL file in VMD main: Plugin -> TK console -> source /full/path/to/your/tcl/file.tcl

# Example Visualization
Below are example snapshots of the VMD model generated using the Tcl script.
1. Community Overview:
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/0db4b392-546e-4738-8e73-6443049e903d" />

2. AlphaIIb Beta3 Model (C1-C2 interaction):
Communities C1 and C2 are highlighted with distinct colors, while the remaining communities are shown in neutral colors to provide clear visualization of interactions between C1 and C2.

You can also explore this in 3D in VMD by loading the example visualization state provided in the scripts folder. Guide: Load the .pdb or .gro file, then the .xtc trajectory, and finally load the visualization state alphaIIb_beta3_C1-C2 interaction.vmd.

<img width="600" height="400" alt="alphaIIb_beta3_C1-C2 interaction" src="https://github.com/user-attachments/assets/757abe96-1e6e-4978-8d98-7e9ef6eba317" />


