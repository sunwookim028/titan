#!/bin/bash
CUDA_VISIBLE_DEVICES=2 ncu --export profile/300bp.2k.B32.profile --target-processes all -f --set full bash bwa-mem-gpu.300bp.2k.sh
