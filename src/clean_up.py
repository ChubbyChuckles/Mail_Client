# trading_bot/src/clean_up.py
import glob
import os
from pathlib import Path
from typing import List
import shutil
import datetime

from .config import logger, IS_GITHUB_ACTIONS


def move_files_to_archive(source_dir: str, archive_dir: str) -> None:
    """
    Safely moves all files from the source directory to the archive directory.
    Only used in local PC environment.

    Args:
        source_dir (str): Path to the source directory containing files to move.
        archive_dir (str): Path to the archive directory where files will be moved.
    """
    try:
        # Convert to Path objects for robust path handling
        source_path = Path(source_dir)
        archive_path = Path(archive_dir)

        # Check if source directory exists and is a directory
        if not source_path.exists():
            logger.error(f"Source directory does not exist: {source_dir}")
            return
        if not source_path.is_dir():
            logger.error(f"Source path is not a directory: {source_dir}")
            return

        # Create archive directory if it doesn't exist
        archive_path.mkdir(parents=True, exist_ok=True)

        # Get all files in the source directory
        files = [f for f in source_path.glob("*") if f.is_file()]

        if not files:
            logger.info(f"No files found in source directory: {source_dir}")
            return

        # Move each file to the archive directory
        moved_files = 0
        for file in files:
            try:
                destination_file = archive_path / file.name
                # Check if file already exists in archive
                if destination_file.exists():
                    # Append timestamp to avoid overwriting
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    destination_file = archive_path / f"{file.stem}_{timestamp}{file.suffix}"
                shutil.move(str(file), str(destination_file))
                logger.info(f"Moved file: {file} to {destination_file}")
                moved_files += 1
            except PermissionError:
                logger.error(f"Permission denied when moving file: {file}")
            except Exception as e:
                logger.error(f"Error moving file {file}: {str(e)}")

        logger.info(f"Moved {moved_files} files from {source_dir} to {archive_dir}")

    except Exception as e:
        logger.error(f"Error processing source directory {source_dir}: {str(e)}")


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
    """
    Performs garbage collection by either deleting old Parquet files (GitHub Actions)
    or moving them to an archive directory (local PC), and cleaning up log directories.
    """
    from . import config  # Import here to avoid circular imports

    # Define the list of directories to process for deletion
    directories = [
        "trading_logs" if not IS_GITHUB_ACTIONS else "/tmp/trading_logs",
    ]

    # Define source directory for Parquet files
    source_dir = config.config.RESULTS_FOLDER  # Uses /tmp/data_1m_pq_alot in GitHub Actions
    archive_dir = r"F:\Crypto_Trading\Market_Data"  # Only used locally

    try:
        if IS_GITHUB_ACTIONS:
            # In GitHub Actions, delete old Parquet files instead of archiving
            logger.info(f"Running garbage collection in GitHub Actions: Deleting old Parquet files in {source_dir}")
            delete_old_files([source_dir], keep_count=5)
        else:
            # Local PC: Move Parquet files to archive and clean up directories
            logger.info(f"Running garbage collection locally: Moving files from {source_dir} to {archive_dir}")
            move_files_to_archive(source_dir, archive_dir)
            # Optionally include archive_dir for cleanup
            # directories.append(archive_dir)
            delete_old_files(directories, keep_count=5)
    except Exception as e:
        logger.error(f"Unexpected error in garbage collection: {str(e)}")