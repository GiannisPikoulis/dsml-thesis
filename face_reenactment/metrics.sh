#!/bin/bash

echo "Input #1: $1"
echo "Input #2: $2"
echo "Running on GPU devices with IDs: $3"

fidelity --gpu $3 \
         --isc \
         --fid \
         --kid \
         --prc \
         --save-cpu-ram \
         --input1 $1 \
         --input2 $2
