# dynamical-system-identification

a bunch of python scripts to:

* excite a robot (using [Yarp](https://github.com/robotology/yarp) at the moment) and record measurements (using a separate C++ [GYM](https://github.com/robotology-playground/GYM) module)
* identify mass, com, and inertia parameters from a URDF model description of the robot and from the previously taken measurements

requirements:

* python 2.7
* python modules: numpy, scipy, matplotlib, ipdb, iDynTree (yarp for excitation)
* c compiler, [robotology-superbuild](https://github.com/robotology-playground/robotology-superbuild) module)


known issues:

* excitation is not very generic yet
* using position control over yarp is suboptimal and has timing issues