import os
from Library import Generate

# --- Define paths ---
# Prompt the user for the GLB file path
while True:
    glb_input_path = input("Please enter the full path to your GLB file: ").strip().strip('"').strip("'")
    glb_file_path = os.path.abspath(os.path.expanduser(glb_input_path))

    if os.path.exists(glb_file_path):
        if glb_file_path.lower().endswith(".glb"):
            break
        else:
            print("Error: The provided file is not a .glb file. Please enter a valid GLB file path.")
    else:
        print(f"Error: File not found at '{glb_file_path}'. Please check the path and try again.")

unityAxisFormat = input("Is the GLB file in Unity's Y-up format? (0/1, default is 0): ") == "1" 
Generate(glb_file_path, unityAxisFormat)