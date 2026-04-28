# Neural Networks for curve fitting + meta-heuristic algorithm for structure optimization
This repository is based on a previously done project: https://github.com/navidgolkar/curve-fitting-nn. 
So the base parameters are like this repository and for avoiding bulky explanation, just the added/changed aspects are explained here:
- in train.py the 'Early stopping' section is removed, and tolerance is checked in optimizer
- --epoch default value is changed to 5000

main.py: In this approach we prune the minimum weight and check if the loss is better, if not we go after the next edge,
and continue this process until checking all edges

main2.py: In this approach we use Grey Wolf Optimization (GWO) algorithm as a meta-heuristic approach for optimization of our CustomNet structure.
The algorithm prunes the edges in order to reach an optimum structure.

___
to use you can clone this repository and install the packages needed in requirements.txt and run main.py

#### The input data formula: $2e^{-x}(\sin(5x)+x\cos(5x))$
___

## To-do:
- combine main.py and main2.py
- add other approaches (random forest, genetic algorithm, etc.)
- tidy up the codes