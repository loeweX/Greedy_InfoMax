echo "Make sure conda is installed."
echo "Installing environment:"
conda env create -f environment.yml || conda env update -f environment.yml || exit
conda activate infomax
