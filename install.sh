#!/usr/bin/env sh
HOME=`pwd`

# Note Chamfer Distance
cd $HOME/extensions/chamfer_dist
python setup.py install --user

# NOTE: KNN
cd $HOME/extensions/
python setup.py install --user
