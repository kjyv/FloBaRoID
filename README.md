# FloBaRoID [![Build Status](https://travis-ci.org/kjyv/FloBaRoID.svg?branch=master)](https://travis-ci.org/kjyv/FloBaRoID)

(FLOating BAse RObot dynamical IDentification)

FloBaRoID is a framework for parameter identification of floating-base rigid body tree-structures such as
humanoid robots. It aims to provide a complete solution to obtain identified parameters from measurements.

Modules:

* excitation: find optimized trajectories and control the robot movement and record the state and torque measurements (using [Yarp](https://github.com/robotology/yarp) or ROS/MoveIt!, probably needs some customization)
* identification: identify dynamical parameters (mass, COM and rotational inertia) starting from an URDF description (providing the kinematic parameters) and from torque and force measurements
    * parameters are constrained to physical consistent standard parameter space, improving robustness against input data that is not well-conditioned
* write the identified parameters into output URDF ready to use for control or simulation

Details:

* finds ideal excitation trajectories with non-linear global optimization (as parameters of fourier-series for periodic soft trajectories) 
* data preprocessing
    * derives velocity and acceleration values from position readings
    * data is zero-phase low-pass filtered from supplied measurements
    * it is possible to only select a combination of data blocks to yield a better condition number than all of the data
* validation with other measurement data
* implemented estimation methods:
  * ordinary least squares, OLS
  * weighted least squares (Zak)
  * estimation of parameter error using previously known CAD values (Gautier)
  * essential standard parameters (Pham, Gautier), estimating only those that are most certain for the measurement data and leaving the others unchanged
  * identification problem formulation with constraints as linear convex SDP problem to get optimal physical consistent parameters (Sousa)

requirements for identification module:

* at least python 2.7 or 3.3
* python modules: numpy (> 1.8), scipy, sympy (>= 1.0), iDynTree, pyyaml, numpy-stl, cvxopt, pylmi-sdp, matplotlib, colorama, palettable, humanize, future (when using Python 2.7)
* dsdp5 (command line executable)
* symengine.py (for SDP speedups)
* mpld3, jinja2 (if using html plots)

requirements for excitation modules:

* optimization: python modules: iDynTree, pyOpt (fork at https://github.com/kjyv/pyOpt is
  recommended)
* for yarp/walkman: c compiler, installed [robotology-superbuild](https://github.com/robotology-playground/robotology-superbuild), python modules: yarp
* for ros/kuka: [kuka-lwr package](https://github.com/CentroEPiaggio/kuka-lwr), python modules: ros, moveit\_msg, moveit\_commander
* for other robots, new modules might have to be written

known issues:

* excitation modules are not generic for any robot
* using position control over yarp is suboptimal and can expose timing issues (seems to happen especially with used python to c bridge)
* COM constraints need stl mesh files for the model to compute the enclosing hull, doesn't e.g. read geometric shape definitions for link

(more or less) Quick start :

* copy one of the existing .yaml configuration files and customize for your setup

* optionally, use the trajectory.py script to generate an optimal exciting trajectory (only fixed
  base at the moment).

* get joint torque measurements from your robotic system if possible e.g. by using the excite.py
  script (filters the measurements as well as gets velocity and acceleration from position
  measurements).  If you are using other means of motion control and data recording, the data files
  of the numpy data files need to have the expected data fields (see README.md in ./excitation/) and
  the data needs to be filtered before running the identification on it. There is also a csv2npz
  script that can be customized to load data from csv text files.

* run identification.py, at least supplying the measurement data file and the corresponding
  kinematic model in a .urdf file with some physically consistent CAD parameters as starting point
  (parameters are not necessary for all methods but recommended).
Optionally you can supply an output .urdf file path to which the input urdf is written with the
identified parameters instead. Another measurements file can be supplied for validation. Most
options are in the .yaml file, others are explained when calling the scripts with --help.


SDP optimization code is based on or uses parts from [cdsousa/wam7\_dyn\_ident](https://github.com/cdsousa/wam7_dyn_ident)

Usage is licensed under the LGPL 3.0, see License.md. Please quote name and authors if you're using this software in any project.
