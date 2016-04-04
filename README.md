# dynamical-system-identification

a bunch of python scripts to:

* excite a robot (using [Yarp](https://github.com/robotology/yarp) at the moment) and record measurements (using a separate C++ [GYM](https://github.com/robotology-playground/GYM) module)
* identify mass, com, and inertia parameters from a URDF model description of the robot and from the previously taken measurements

details:
* excitation can be parameterized to get ideal trajectories (finding these parameters not yet
  implemented)
* acceleration and velocity values can be derived from position readings, all are low-pass filtered without time shift
* implements reduction of standard parameters to base parameters and further to essential
  parameters, allows estimating only those
* allows weighted least squares instead of ordinary least squares
* allows estimation of parameter error in addition to absolute parameters using known CAD values

requirements for identification:
* python 2.7
* python modules: numpy, scipy, matplotlib, ipython, ipdb, iDynTree python bindings, colorama

requirements for excitation:
* c compiler, installed [robotology-superbuild](https://github.com/robotology-playground/robotology-superbuild) module)
* python modules: yarp or ros, moveit_msg, moveit_commander

known issues:
* excitation is not very generic yet
* using position control over yarp is suboptimal and can expose timing issues
