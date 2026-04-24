# Neural Networks for curve fitting + meta-heuristic algorithm for structure optimization
This repository is based on a previously done project: https://github.com/navidgolkar/curve-fitting-nn. 
So the base parameters are like this repository and for avoiding bulky explanation, just the added aspects are explained here:
- --tol default value is changed to 1e-3
- --epoch default value is changed to 5000

In this repository we use Grey Wolf Optimization (GWO) algorithm as a meta-heuristic approach for optimization of our CustomNet structure.
The algorithm prunes the edges in order to reach an optimum structure.

___
to use you can clone this repository and install the packages needed in requirements.txt and run main.py

#### The input data formula: $2e^{-x}(\sin(5x)+x\cos(5x))$
___

## To-do:
- make GWO_parameters.py for defining hyper parameters of optimizer
- implement the seed for optimizer in GWO_parameters.py as well
- implement computing edge importance, applying mask, and gwo_run loop