#!/usr/bin/env bash

# Standard packages
pip install numpy
pip install scipy
pip install pandas
pip install matplotlib
pip install seaborn

# Neural network packages
pip install tensorflow==2.14
pip install tensorflow-probability==0.22.1

# Specialized packages
pip install biopython
pip install deeplift
pip install editdistance
pip install logomaker
pip install prtpy
pip install tables

# Explainable AI package, custom version
git clone https://github.com/castillohair/shap
cd shap
python setup.py develop
cd ..
    
# AI sequence design packages
git clone https://github.com/castillohair/isolearn
cd isolearn
python setup.py develop
cd ..

git clone https://github.com/castillohair/genesis
cd genesis
python setup.py develop
cd ..

git clone https://github.com/castillohair/corefsp
cd corefsp
python setup.py develop --no-deps
cd ..
