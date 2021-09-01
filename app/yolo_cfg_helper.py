from typing import List
import helper
import os
from glob import glob
import re
import subprocess
import logging


def replace_classes_and_filters(classes_count: int, training_folder: str) -> None:
    cfg_file = _find_cfg_file(training_folder)

    with open(cfg_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith('filters='):
            last_known_filters_line = i
        if line.startswith('[yolo]'):
            lines[last_known_filters_line] = f'filters={(classes_count+5)*3}\n'
            last_known_filters_line = None
        if line.startswith('classes='):
            lines[i] = f'classes={classes_count}\n'

    with open(cfg_file, 'w') as f:
        f.writelines(lines)


def update_anchors(training_folder: str) -> None:
    cfg_file_path = _find_cfg_file(training_folder)
    yolo_layer_count = _read_yolo_layer_count(cfg_file_path)
    width, height = _read_width_and_height(cfg_file_path)

    anchors = _calculate_anchors(training_folder, yolo_layer_count, width, height)
    _write_anchors(cfg_file_path, anchors)


def update_hyperparameters(
        training_folder: str,
        batch: int = 64,
        subdivisions: int = 8,
        size: int = 800,
        learning_rate: int = 0.001,
        burn_in: int = 400,
        steps: List[int] = [30000, 50000],
        max_batches=80000
) -> None:
    cfg = _find_cfg_file(training_folder)
    with open(cfg, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith("batch"):
            lines[i] = f'batch={batch}\n'
        if line.startswith("subdivisions"):
            lines[i] = f'subdivisions={subdivisions}\n'
        if line.startswith("width"):
            lines[i] = f'width={size}\n'
        if line.startswith("height"):
            lines[i] = f'height={size}\n'
        if line.startswith("learning_rate"):
            lines[i] = f'learning_rate={learning_rate}\n'
        if line.startswith("burn_in"):
            lines[i] = f'burn_in={burn_in}\n'
        if line.startswith("steps"):
            lines[i] = f'steps={",".join(steps)}\n'
        if line.startswith("max_batches"):
            lines[i] = f'max_batches={max_batches}\n'
        if line.startswith("[convolutional]"):
            break

    logging.info(''.join(lines[:30]))

    with open(cfg, 'w') as f:
        f.writelines(lines)


def _find_cfg_file(folder: str) -> str:
    cfg_files = [file for file in glob(f'{folder}/**/*', recursive=True) if file.endswith('.cfg')]
    if len(cfg_files) == 0:
        raise Exception(f'[-] Error: No cfg file found.')
    elif len(cfg_files) > 1:
        raise Exception(f'[-] Error: Found more than one cfg file: {cfg_files}')
    return cfg_files[0]


def _calculate_anchors(training_path, yolo_layer_count: int, width: int, height: int):
    cmd = f'cd {training_path};/darknet/darknet detector calc_anchors data.txt -num_of_clusters {yolo_layer_count*3} -width {width} -height {height}'
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if (p.returncode != 0):
        raise Exception(f'Calculating anchors failed:\nout: {out.decode("utf-8")} \nerror: {err.decode("utf-8") }')
    with open(f'{training_path}/anchors.txt', 'r') as f:
        anchors = f.readline()
    os.remove(f'{training_path}/anchors.txt')
    return anchors


def _read_yolo_layer_count(cfg_file_path: str):
    with open(cfg_file_path, 'r') as f:
        lines = f.readlines()

    yolo_layers = [line for line in lines
                   if line.lower().startswith('[yolo]')]
    return len(yolo_layers)


def _read_width_and_height(cfg_file_path: str):
    with open(cfg_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("width="):
            width = re.findall(r'\d+', line)[0]

        if line.startswith("height="):
            height = re.findall(r'\d+', line)[0]
    return width, height


def _write_anchors(cfg_file_path: str, anchors: str) -> None:
    with open(cfg_file_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("anchors"):
            lines[i] = f'anchors={anchors}\n'
    with open(cfg_file_path, 'w') as f:
        f.writelines(lines)
