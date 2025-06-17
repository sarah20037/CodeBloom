import os
import sys
import shutil
from fawkes.protection import protect

# ========== Configuration ==========
INPUT_IMAGE = r"C:\Users\hp\codebloom\photo.jpeg"
OUTPUT_DIR = r"C:\Users\hp\codebloom\fawkes_output"
MODE = "high"  # Options: "low", "medium", "high", "ultra"
MODEL = "facenet"  # Options: "facenet", "arcface"

# ========== Prepare Input ==========
temp_input_dir = "./temp_input"
if os.path.exists(temp_input_dir):
    shutil.rmtree(temp_input_dir)
os.makedirs(temp_input_dir, exist_ok=True)

shutil.copy(INPUT_IMAGE, temp_input_dir)

# ========== Run Fawkes Protection ==========
protect(
    input_dir=temp_input_dir,
    output_dir=OUTPUT_DIR,
    mode=MODE,
    model=MODEL,
    format="png"
)

# ========== Cleanup ==========
shutil.rmtree(temp_input_dir)
