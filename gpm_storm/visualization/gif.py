#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:04:07 2023

@author: ghiggi
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np

def check_input_files(image_filepaths):
    """Check valid input file paths."""
    # Check at least 2 files
    if not isinstance(image_filepaths, list):
        raise ValueError("Expecting a list.")
    if len(image_filepaths) < 2:
        raise ValueError("Expecting a list 2 file paths")

    # Check file exist
    if np.any([not os.path.exists(fpath) for fpath in image_filepaths]):
        raise ValueError("Not all file paths exists on disk.")

    # Check file format
    # TODO

    return image_filepaths


def check_frame_settings(frame_duration, frame_rate, return_duration=False):
    if frame_duration is not None and frame_rate is not None:
        raise ValueError("Either specify frame_duration or frame_rate.")

    if frame_duration is None and frame_rate is None:
        frame_rate = 4

    if frame_duration is not None:
        frame_rate = int(1000 / frame_duration)
    else:
        frame_duration = int(1000 / frame_rate)
    if return_duration:
        return frame_duration
    else:
        return frame_rate


def _move_and_rename_image_to_tmp_dir(filepaths, tmp_dir=None, delete_inputs=False):
    # Create a temporary directory
    tmp_dir = tempfile.mkdtemp(prefix="tmp_images_", dir=tmp_dir)
    pattern = "image_{:04d}.png"
    # Copy files to the temporary directory and rename them
    # - Do not move because a single image can be referenced multiple times in filepaths
    for i, filepath in enumerate(filepaths):
        _, filename = os.path.split(filepath)
        new_filename = pattern.format(i + 1)  # Start index from 1
        new_filepath = os.path.join(tmp_dir, new_filename)
        shutil.copy(filepath, new_filepath)

    # Delete inputs
    if delete_inputs:
        for filepath in filepaths:
            try:
                os.remove(filepath)
            except FileNotFoundError:
                pass

    pattern = "image_%04d.png"
    return tmp_dir, pattern


def get_image_size(image_path):
    from PIL import Image

    img = Image.open(image_path)
    width, height = img.size
    return width, height


def create_gifski_gif(
    image_filepaths,
    gif_fpath,
    sort=False,
    frame_duration=None,
    frame_rate=None,
    loop=0,
    quality=100,
    motion_quality=100,
    lossy_quality=100,
    optimize=True,
    delete_inputs=False,
    verbose=True,
):
    """
    Create a GIF from a list of image filepaths using gifski.

    Gifski generates per-frame optimized palettes, combines palettes across frames
    and achieve thousands of colors per frame for maximum quality.

    Either specify frame_rate or frame_duration.

    Parameters
    ----------
    image_filepaths : list of str
        List of filepaths of input images. The images will be used to create the GIF.
    gif_fpath : str
        Filepath of the output GIF.
    sort : bool, optional
        If True, sort the input images in ascending order. Default is False.
    frame_duration : int, optional
        Duration in milliseconds (ms) of each GIF frame.
        The default is 250 ms.
    frame_rate: int, optional
        The number of individual frames displayed in one second.
        The default is 4.
    quality: str, optional
        Controls the overall output quality. The default is 100.
        Lower values reduces quality but may reduce file size.
    motion_quality: str, optional
        Controls motion quality. The default is 100.
        Lower values reduce motion quality.
    lossy_quality: str, optional
        Controls PNG quantization quality. The default is 100.
        Lower values introduce noise and streaks.
    optimize: bool, optional
        If True, it improves the GIF quality (--extra argument).
        If False, it speeds up the GIF creation at quality expense.
    loop : int, optional
        Number of times the GIF should loop. Set to 0 for infinite loop. Default is 0.
    delete_inputs : bool, optional
        If True, delete the original input images after creating the GIF. Default is False.

    Notes
    -----
    This function uses gifski to create the GIF. Ensure that gifski is installed and
    accessible in the system's PATH.

    On Linux systems, it can be installed using 'sudo snap install gifski'.

    More GIFski information at: https://github.com/ImageOptim/gifski

    More PNG quantization information at: https://pngquant.org/

    Examples
    --------
    >>> filepaths = ["image1.png", "image2.png", "image3.png"]
    >>> gif_fpath = "output.gif"
    >>> create_gifski_gif(filepaths, gif_fpath, sort=True, frame_duration=200)
    """
    # Define frame rate
    frame_rate = check_frame_settings(frame_duration, frame_rate)

    # Sort image filepaths if required
    if sort:
        image_filepaths.sort()

    # Retrieve image width and height
    width, height = get_image_size(image_filepaths[0])

    # Move images to a temporary directory
    # - gifski is not able to process image that are not in /home/*
    # --> I can not use /tmp or /ltenas
    base_dir = os.path.join(os.path.expanduser("~"), "tmp_gifski")  # /home/<user>/tmp_gifski
    os.makedirs(base_dir, exist_ok=True)
    tmp_dir, pattern = _move_and_rename_image_to_tmp_dir(image_filepaths, tmp_dir=base_dir, delete_inputs=delete_inputs)
    input_pattern = os.path.join(tmp_dir, "image_*.png")
    tmp_fpath = os.path.join(base_dir, os.path.basename(gif_fpath))

    # Define basic command
    gifski_cmd = [
        "gifski",
        # Do not sort files
        "--no-sort",
        # Framerate
        "--fps",
        str(frame_rate),  # input framerate. must be greater or equal to fps
        # Set output quality
        "--quality",
        str(quality),
        "--motion-quality",
        str(motion_quality),
        "--lossy-quality",
        str(lossy_quality),
        # Loops
        "--repeat",
        str(loop),
        # GIF size
        "--width",
        str(width),
        "--height",
        str(height),
    ]

    # Add optimization option if specified
    if optimize:
        gifski_cmd.extend(["--extra"])

    # Add output filepath
    # - Overwrite existing !
    gifski_cmd.extend(["--output", tmp_fpath])

    # Add input file paths pattern
    gifski_cmd.extend([input_pattern])

    # Define the command
    gifski_cmd = " ".join(gifski_cmd)

    print(gifski_cmd)
    # Run gifski using subprocess
    try:
        subprocess.run(gifski_cmd, check=True, shell=True, capture_output=not verbose)
        shutil.move(tmp_fpath, gif_fpath)
    except subprocess.CalledProcessError as e:
        print(f"Error creating GIF: {e}")

    # Remove temporary directory and its contents
    shutil.rmtree(tmp_dir)