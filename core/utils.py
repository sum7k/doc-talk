import os, re


def normalize_filename(filename: str) -> str:
    """Normalize filename by removing/replacing problematic characters."""
    # Get the name and extension
    name, ext = os.path.splitext(filename)
    
    # Replace problematic characters with underscores
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Replace multiple spaces with single underscore
    name = re.sub(r'\s+', '_', name)
    # Remove leading/trailing dots and spaces
    name = name.strip('. ')
    # Ensure it's not empty
    if not name:
        name = 'unnamed_file'
    
    return f"{name}{ext}"