#!/bin/bash

if [ "$#" -lt 1 ]
then
    echo "Converts ipython notebook files to python."  
    echo "Usage: nb2py.sh 1.ipynb 2.ipynb ..."
    echo ""
    echo "Description:"
    echo "Use this routine to convert ipython notebook files to python files.  This is useful for storing these files in git.  Storing ipython notebook files into a git repo is not ideal. ipython notebooks also contain output cells that can potentially change even when the code (or comments) have not been modified.  This renders tracking changes tedious."
fi
    
for file in "$@"
do
    if [[ $file =~ \.ipynb$ ]]
    then
        echo "[Processing]  " $file
        jupyter nbconvert --to=python $file
        echo "[Cleaning up] " "$(basename "$file" .ipynb).py" 
        sed -i .nb2py_bak '/^# In\[[ 0-9]*\]/d' "$(basename "$file" .ipynb).py"
        rm -f "$(basename "$file" .ipynb).py".nb2py_bak
        echo "[Deleting]    " "$(basename "$file" .ipynb).py".nb2py_bak
    else
        echo "Ignoring   - "$file
    fi
done

 
