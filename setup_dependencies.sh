#!/usr/bin/env bash
echo "Make sure conda is installed."
echo "Installing environment:"
conda env create -f environment.yml || exit
conda activate infomax
