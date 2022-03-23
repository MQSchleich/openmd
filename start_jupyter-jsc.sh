#!/bin/bash
# this script (start_jupyter-jsc.sh) must be in the default $HOME/.jupyter to be used by Jupyter-JSC

module purge
module load Stages/2022
module load GCCcore/.11.2.0
module load PyCUDA/2021.1

module use /p/usersoftware/swmanage/goebbert1/stage2022/jupyter33/easybuild/$SYSTEMNAME/modules/all/Compiler/GCCcore/11.2.0/

module load Python
module load JupyterCollection/2022.3.3
