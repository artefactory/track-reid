#!/bin/bash -e

read -p "Want to install conda env named 'track-reid'? (y/n)" answer
if [ "$answer" = "y" ]; then
  echo "Installing conda env..."
  conda create -n track-reid python=3.10 -y
  source $(conda info --base)/etc/profile.d/conda.sh
  conda activate track-reid
  echo "Installing requirements..."
  pip install -r requirements-developer.txt
  python3 -m ipykernel install --user --name=track-reid
  conda install -c conda-forge --name track-reid notebook -y
  echo "Installing pre-commit..."
  make install_precommit
  echo "Installation complete!";
else
  echo "Installation of conda env aborted!";
fi
