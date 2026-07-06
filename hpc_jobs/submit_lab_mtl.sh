#!/bin/bash
# Submit MTL training array job for all LAB_DDM_paper experiments (task IDs 0-8).
# Assumes standard 03_main_training.py has already been run (existing
# classification_performances*.joblib files will be loaded and MTL keys merged in).
#
# Usage:
#   bash submit_lab_mtl.sh            # submit all 9 tasks
#   bash submit_lab_mtl.sh 0-2        # submit only tasks 0, 1, 2 (concentration folders)
#
# Concentration folders only (recommended for meaningful regression):
#   Task 0: ACA_qdPCR
#   Task 1: AMCA_qdLAMP
#   Task 2: AMCA_qdPCR
# Other folders (Z_area, Z_range, etc.) will run with sentinel-only concentration
# (regression loss = 0 for all samples; classification still trains normally).

RANGE=${1:-0-8}

echo "Submitting MTL training job array: tasks ${RANGE}"
qsub -J "${RANGE}" /rds/general/user/gk225/home/POC_DDM/hpc_jobs/lab_mtl_training.pbs
