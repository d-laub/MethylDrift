# MethylDrift

***updates and revisions to code-base still being made**

## Installation
See the `dependencies.yml` for dependencies. Once dependencies are met, clone the git repository and add the repo to your environment's `PYTHONPATH` variable. Installation via `pip` or Anaconda might be implemented in the future.

For example, assuming you are in a bash terminal with Anaconda installed, you would navigate to where you want to install MethylDrift and enter:
```bash
# Clone repo
git clone https://github.com/d-laub/MethylDrift.git

# Create conda environment with dependencies installed.
cd MethylDrift
conda env create -f dependencies.yml -n MethylDrift

# Add the directory to your Python path
cd ..
echo "export PYTHONPATH=${PWD}${PYTHONPATH:+:${PYTHONPATH}}" >> ~/.bashrc
source ~/.bashrc
```

