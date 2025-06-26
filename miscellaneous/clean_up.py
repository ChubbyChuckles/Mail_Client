import os
import glob
from pathlib import Path
from typing import List
from src.config import logger

def delete_old_files(directories: List[str], keep_count: int = 5) -> None:
    """
    Deletes all files in each directory except the keep_count most recently modified files.

    Args:
        directories (List[str]): List of directory paths to process
        keep_count (int): Number of most recent files to keep (default: 5)
    """
    for directory in directories:
        try:
            # Convert to Path object for robust path handling
            dir_path = Path(directory)

            # Check if directory exists
            if not dir_path.exists():
                logger.error(f"Directory does not exist: {directory}")
                continue
            if not dir_path.is_dir():
                logger.error(f"Path is not a directory: {directory}")
                continue

            # Get all files in the directory
            files = [f for f in dir_path.glob("*") if f.is_file()]

            if not files:
                logger.info(f"No files found in directory: {directory}")
                continue

            # Sort files by modification time (most recent first)
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Keep only the latest keep_count files
            files_to_delete = files[keep_count:]

            if not files_to_delete:
                logger.info(f"Directory {directory} has {len(files)} files, all will be kept (less than or equal to {keep_count})")
                continue

            # Delete older files
            for file in files_to_delete:
                try:
                    file.unlink()
                    logger.info(f"Deleted file: {file}")
                except PermissionError:
                    logger.error(f"Permission denied when deleting file: {file}")
                except Exception as e:
                    logger.error(f"Error deleting file {file}: {str(e)}")

            logger.info(f"Directory {directory}: Kept {min(keep_count, len(files))} files, deleted {len(files_to_delete)} files")

        except Exception as e:
            logger.error(f"Error processing directory {directory}: {str(e)}")

def garbage_collection():
    # Define the list of directories to process
    directories = [
        "trading_logs",
        "data_1m_pq_alot",
        # "/path/to/directory3"
    ]

    try:
        delete_old_files(directories, keep_count=5)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")