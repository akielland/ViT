SHELL := /bin/bash

run:
    module purge && \
    module load JupyterLab/3.5.0-GCCcore-11.3.0 && \
    source /fp/projects01/ec232/venvs/in5310/bin/activate && \
    python p1_ViT.py && \
    python p1_ViT_LoRA_02.py

