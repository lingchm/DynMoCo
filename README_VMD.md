# Objective
This project provides a Python script that generates a Tcl script for VMD to visualize dynamic community assignments of residues over time. Given a JSON file containing residue assignments to communities across multiple timestamps, the script enables users to color-code residues according to their community membership for each frame of a trajectory.

Each community is annotated with a number directly in the VMD model, positioned near the corresponding residues, enabling communities to be visually distinguished and tracked throughout the trajectory. This dynamic visualization facilitates direct observation of structural rearrangements over time and supports detailed analysis of both local residue interactions and global conformational changes.

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

