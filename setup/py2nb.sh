#!/bin/bash

if [ "$#" -lt 1 ]
then
    echo "Converts python files to ipython notebooks."  
    echo "Usage: py2nb.sh 1.py 2.py ..."
    echo ""
    echo "Requires python p2j package (pip3 install p2j)."
    echo ""
    echo "Description:"
    echo "It is possible to convert ipython notebook files to python files using \"jupyter nbconvert --to=python file.ipynb\".  This is useful when storing files in a git repo.  Storing ipython notebook files into a git repo is not ideal. ipython notebooks also contain output cells that can potentially change even when the code (or comments) have not been modified.  This renders tracking changes tedious."
    echo ""
    echo "When an ipython notebook file is converted into a python file, the output information is stripped away.  The comments section start with a \"# ##\" tag.  The code section is copyied verbatim.  The code section also include cell identification tags of the form \"# In[ ]\".  These cell identification tags need to be removed from the python file before it can be converted back into an ipython notebook.  This script does just that."    
    echo ""
    echo "Example Python File (which can be converted into an ipython notebook using this program):"
    echo "----------------"
    echo "# ## Test data"
    echo "# Creating some test data"
    echo ""
    echo "# In[2]:"
    echo "from sklearn import datasets"
    echo "x, y = datasets.make_moons(n_samples=10, random_state=0, noise=0.1)"
    echo "x = x - np.mean(x,0) # 0 centered"
    echo "----------------"
    echo ""
    exit
fi

for file in "$@"
do
    if [[ $file =~ \.py$ ]]
    then
        echo "Converting - " $file
        sed -i .py2nb_bak '/^# In\[[ 0-9]*\]/d' $file
        p2j $file
    else
        echo "Ignoring   - "$file
    fi
done
