import base64
import os
from argparse import Namespace
from pathlib import Path

from tqdm import tqdm


def encode_image(image_path: str) -> str:
    """
    Encode an image file to a base64 string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image.

    Example:
        >>> encode_image("path/to/image.jpg")
        'iVBORw0KGgoAAAANSUhEUgAA...'
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def strip_trailing_slash(dir: str) -> str:
    """
    Remove trailing slashes from a path.
    
    Args:
        dir (str): Directory path.
    
    Returns:
        str: Directory path without trailing slashes.
    
    Example:
        >>> strip_trailing_slash("path/to/dir/")
        "path/to/dir"
    """
    return dir.rstrip('/')

def get_path_prefix(path: str) -> str:
    """
    Get the parent directory of a given path.

    Args:
        path (str): Path to a file or directory.

    Returns:
        str: Parent directory path.

    Example:
        >>> get_path_prefix("path/to/file.txt")
        'path/to'
    """
    return str(Path(path).parent)

def get_path_suffix(path: str) -> str:
    """
    Get the suffix (extension) of a given path.

    Args:
        path (str): Path to a file.

    Returns:
        str: File suffix (e.g., '.txt').

    Example:
        >>> get_path_suffix("path/to/file.txt")
        '.txt'
    """
    return Path(path).suffix

def get_filename(path: str) -> str:
    """
    Get the filename from a given path.

    Args:
        path (str): Path to a file.

    Returns:
        str: Filename (e.g., 'file.txt').

    Example:
        >>> get_filename("path/to/file.txt")
        'file.txt'
    """
    return Path(path).name

def get_filename_prefix(path: str) -> str:
    """
    Get the filename without its suffix.

    Args:
        path (str): Path to a file.

    Returns:
        str: Filename without suffix (e.g., 'file').

    Example:
        >>> get_filename_prefix("path/to/file.txt")
        'file'
    """
    return Path(path).stem

def get_filename_suffix(path: str) -> str:
    """
    Get the suffix (extension) of a filename.

    Args:
        path (str): Path to a file.

    Returns:
        str: File suffix (e.g., '.txt').

    Example:
        >>> get_filename_suffix("path/to/file.txt")
        '.txt'
    """
    return Path(path).suffix

def change_path_suffix(path: str, new_suffix: str) -> str:
    """
    Change the suffix of a given path.

    Args:
        path (str): Path to a file.
        new_suffix (str): New suffix (e.g., '.jpg').

    Returns:
        str: Updated path with the new suffix.

    Example:
        >>> change_path_suffix("path/to/file.txt", ".jpg")
        'path/to/file.jpg'
    """
    p = Path(path)
    return str(p.with_suffix(new_suffix))

def change_filename_prefix(path: str, new_prefix: str) -> str:
    """
    Change the prefix of a filename while keeping the suffix unchanged.

    Args:
        path (str): Path to a file.
        new_prefix (str): New filename prefix.

    Returns:
        str: Updated path with the new prefix.

    Example:
        >>> change_filename_prefix("path/to/file.txt", "new_")
        'path/to/new_file.txt'
    """
    p = Path(path)
    return str(p.with_name(new_prefix + p.suffix))

def change_filename_suffix(path: str, new_suffix: str) -> str:
    """
    Change the suffix of a filename while keeping the prefix unchanged.

    Args:
        path (str): Path to a file.
        new_suffix (str): New filename suffix.

    Returns:
        str: Updated path with the new suffix.

    Example:
        >>> change_filename_suffix("path/to/file.txt", ".jpg")
        'path/to/file.jpg'
    """
    p = Path(path)
    return str(p.with_name(p.stem + new_suffix))

def change_filename(path: str, new_name: str) -> str:
    """
    Change the full filename (including prefix and suffix).

    Args:
        path (str): Path to a file.
        new_name (str): New filename (e.g., 'new_file.txt').

    Returns:
        str: Updated path with the new filename.

    Example:
        >>> change_filename("path/to/file.txt", "new_file.jpg")
        'path/to/new_file.jpg'
    """
    p = Path(path)
    return str(p.with_name(new_name))

def confirm_restart_if_exists(args: Namespace, completed: set) -> set:
    """Check for existing progress and ask user whether to restart."""
    if not completed:
        return completed

    while True:
        user_input = input(
            f"\nDetected existing completed data ({len(completed)} items) in '{args.outfile}'.\n"
            "Do you want to delete previous results and restart? (Y/N): "
        ).strip().lower()

        if user_input == "y":
            try:
                for fpath in [args.outfile, args.translatefile, args.failfile, args.refinefile]:
                    if os.path.exists(fpath):
                        os.remove(fpath)
                        tqdm.write(f"Deleted '{fpath}'.")
                tqdm.write("Previous data deleted. Starting fresh.")
            except Exception as e:
                tqdm.write(f"Failed to delete files: {e}")
                raise e
            return set()
        elif user_input == "n":
            tqdm.write("Continuing with existing progress.")
            return completed
        else:
            tqdm.write("Invalid input. Please enter 'Y' or 'N'.")
