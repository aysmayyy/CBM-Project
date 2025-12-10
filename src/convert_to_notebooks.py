"""
Convert Python files to Jupyter Notebooks
Keeps all content exactly the same - just wraps in notebook format
"""

import json
import os
import re

def py_to_notebook(py_file, output_dir=None):
    """Convert a .py file to .ipynb notebook"""
    
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into cells by looking for patterns
    # Each function, class, or major comment block becomes a cell
    
    lines = content.split('\n')
    
    cells = []
    current_cell = []
    
    for i, line in enumerate(lines):
        # Start new cell on major patterns
        if (line.startswith('def ') or 
            line.startswith('class ') or
            line.startswith('# ===') or
            line.startswith('"""') and len(current_cell) > 5 or
            line.startswith('if __name__')):
            
            # Save current cell if it has content
            if current_cell and any(l.strip() for l in current_cell):
                cells.append('\n'.join(current_cell))
            current_cell = [line]
        else:
            current_cell.append(line)
    
    # Don't forget last cell
    if current_cell and any(l.strip() for l in current_cell):
        cells.append('\n'.join(current_cell))
    
    # If we only got 1-2 cells, just make it one big cell (simple script)
    if len(cells) <= 2:
        cells = [content]
    
    # Build notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    for cell_content in cells:
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell_content.split('\n')
        })
    
    # Fix source format (needs to have \n at end of each line except last)
    for cell in notebook["cells"]:
        source = cell["source"]
        cell["source"] = [line + '\n' if i < len(source)-1 else line 
                         for i, line in enumerate(source)]
    
    # Output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.basename(py_file).replace('.py', '.ipynb')
        output_path = os.path.join(output_dir, base)
    else:
        output_path = py_file.replace('.py', '.ipynb')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✓ {os.path.basename(py_file)} → {os.path.basename(output_path)}")
    return output_path


def convert_all(input_dir, output_dir):
    """Convert all .py files in a directory"""
    
    py_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.py')])
    
    print(f"Found {len(py_files)} Python files\n")
    
    for py_file in py_files:
        full_path = os.path.join(input_dir, py_file)
        py_to_notebook(full_path, output_dir)
    
    print(f"\n✓ Done! Notebooks saved to: {output_dir}")


if __name__ == "__main__":
    # Convert all .py files in src/ to notebooks/
    import sys
    
    if len(sys.argv) >= 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
    else:
        # Default: current directory → notebooks/
        input_dir = "."
        output_dir = "../notebooks"
    
    convert_all(input_dir, output_dir)