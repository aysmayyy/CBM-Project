#!/usr/bin/env python3
"""
NULL BYTE REMOVER - Fix corrupted files once and for all!
Usage: python fix_null_bytes.py <file_or_directory>
"""

import sys
import os
from pathlib import Path

def remove_null_bytes(file_path):
    """Remove null bytes from a single file"""
    try:
        # Read file in binary mode
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Check if there are null bytes
        null_count = content.count(b'\x00')
        
        if null_count == 0:
            print(f"✓ {file_path} - No null bytes found")
            return True
        
        # Remove null bytes
        cleaned_content = content.replace(b'\x00', b'')
        
        # Write back to the same file
        with open(file_path, 'wb') as f:
            f.write(cleaned_content)
        
        print(f"✓ {file_path} - Removed {null_count} null bytes")
        return True
        
    except Exception as e:
        print(f"✗ {file_path} - Error: {e}")
        return False

def process_path(path):
    """Process a file or directory"""
    path = Path(path)
    
    if not path.exists():
        print(f"Error: {path} does not exist!")
        return
    
    if path.is_file():
        # Process single file
        remove_null_bytes(path)
    
    elif path.is_dir():
        # Process all Python files in directory
        python_files = list(path.glob('**/*.py'))
        
        if not python_files:
            print(f"No Python files found in {path}")
            return
        
        print(f"Found {len(python_files)} Python files\n")
        
        success_count = 0
        for file in python_files:
            if remove_null_bytes(file):
                success_count += 1
        
        print(f"\nProcessed {success_count}/{len(python_files)} files successfully")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_null_bytes.py <file_or_directory>")
        print("\nExamples:")
        print("  python fix_null_bytes.py my_script.py")
        print("  python fix_null_bytes.py ./my_project")
        print("  python fix_null_bytes.py .")
        sys.exit(1)
    
    path = sys.argv[1]
    print(f"=== NULL BYTE REMOVER ===\n")
    process_path(path)

if __name__ == "__main__":
    main()