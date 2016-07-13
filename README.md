# dynamical-system-identification

* excite a robot with fourier series trajectories and record state and torque measurements (using [Yarp](https://github.com/robotology/yarp) or ROS/MoveIt! at the moment)
* identify mass, com, and inertia parameters using a URDF model description of the robot and from the measurements

details:
* excitation can be parameterized to get ideal trajectories (finding these parameters by optimization not yet
  implemented)
* acceleration and velocity values can be derived from position readings, all are low-pass filtered without time shift
* implements reduction of standard parameters to base parameters and further to essential parameters, estimating only those that are relevant for the measurement data and leaving the others untouched
* allows weighted least squares instead of ordinary least squares
* allows estimation of parameter error in addition to absolute parameters using previously known CAD values
* save identified values back to URDF

requirements for identification:
* python 2.7
* python modules: numpy, scipy, sympy, matplotlib, iDynTree, colorama, humanize, (ipython, ipdb),
  pylmi-sdp, cvxopt
* optionally for output as html: mpld3, jinja2

requirements for excitation:
* for yarp/walkman: c compiler, installed [robotology-superbuild](https://github.com/robotology-playground/robotology-superbuild) module, python modules: yarp
* for ros/kuka: [kuka-lwr package](https://github.com/CentroEPiaggio/kuka-lwr), python modules: ros, moveit_msg, moveit_commander

known issues:
* excitation methods could be more generic, generate optimized trajectories
* using position control over yarp is suboptimal and can expose timing issues
