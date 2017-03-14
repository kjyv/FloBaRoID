# FloBaRoID [![Build Status](https://travis-ci.org/kjyv/FloBaRoID.svg?branch=master)](https://travis-ci.org/kjyv/FloBaRoID)

(FLOating BAse RObot dynamical IDentification)

FloBaRoID is a python toolkit for parameter identification of floating-base rigid body tree-structures such as
humanoid robots. It aims to provide a complete solution for obtaining physical consistent identified dynamics parameters.

Modules:

* **trajectory.py**: generate optimized trajectories
* **visualize.py**: show trajectory with 3D robot model 
* **excite.py**: send trajectory to control the robot movement and record the resulting measurements, using ROS/MoveIt! or [Yarp](https://github.com/robotology/yarp) (probably needs some customization)
* **identify.py**: identify dynamical parameters (mass, COM and rotational inertia) starting from an URDF description (providing the kinematic parameters) and from torque and force measurements


Features:

* find optimized excitation trajectories with non-linear global optimization (as parameters of fourier-series for periodic soft trajectories) 
* data preprocessing
    * derives velocity and acceleration values from position readings
    * data is zero-phase low-pass filtered from supplied measurements
    * it is possible to only select a combination of data blocks to yield a better condition number
* validation with other measurement files
* implemented estimation methods:
  * ordinary least squares, OLS
  * weighted least squares (Zak, 1994)
  * estimation of parameter error using previously known CAD values (Gautier, 2013)
  * essential standard parameters (Pham, Gautier, 2013), estimating only those that are most certain for the measurement data and leaving the others unchanged
  * identification problem formulation with constraints as linear convex SDP problem to get optimal physical consistent standard parameters (Sousa, 2014)
  * non-linear optimization within consistent parameter space (Traversaro, 2016)
* visualization
* plotting
* write the identified parameters directly into output URDF

requirements for identification module:

* python 2.7 or >=3.3
* python modules
	* numpy (> 1.8), scipy, sympy (>= 1.0), iDynTree, pyyaml, numpy-stl, cvxopt, pylmi-sdp, matplotlib (>= 1.4), colorama, palettable, humanize, tqdm
	* when using Python 2.7: future
	* when using Python <3.5: typing

optional:  

* symengine.py (to speedup SDP)
* mpld3, jinja2 (for html plots)
* matplotlib2tikz (for tikz plots)
* rbdl (alternative for inverse dynamics)
* pyglet, pyOpenGL (for visualizer)
* dsdp5 (command line executable)

requirements for excitation module:

* for ros, python modules: ros, moveit\_msg, moveit\_commander
* for yarp (e.g. walkman): c compiler, installed [robotology-superbuild](https://github.com/robotology-playground/robotology-superbuild), python modules: yarp
* for other robots, new modules might have to be written

requirements for optimization module:

* optimization: python modules: iDynTree, pyOpt (fork at https://github.com/kjyv/pyOpt is recommended)
* pyipopt from https://github.com/xuy/pyipopt (plus cmd line ipopt/libipopt)
* mpi4py / mpirun (for parallel trajectory optimization)


known issues:

* trajectory optimization is limited to fixed-base robots ATM
* trajectory optimization does not yet check for self or other collisions (e.g. objects from a world URDF)
* visualizer does not use stl mesh files (so far only bounding boxes)
* visualizer does not yet display trajectories, only statis postures
* YARP excitation module is not generic (ROS should be)
* using position control over yarp is suboptimal and can expose timing issues (seems to happen especially with used python to c bridge)
* COM constraints need an stl mesh files for the model to compute the enclosing hull, doesn't e.g. read geometric shape definitions for link
* non-linear solving not yet activated for identification, buggy?

Quick start tutorial:

* copy one of the existing .yaml configuration files and customize for your setup

* optionally, use the trajectory.py script to generate an optimal exciting trajectory (only fixed base at the moment).

* get joint torque measurements from your robotic system, if suitable e.g. by using the excite.py
  script (including filtering of measurements as well as velocity and acceleration derivation). If you are using other means of motion control and data recording, the data files of the numpy data files need to have the expected data fields (see README.md in ./excitation/) and the data needs to be filtered before running the identification on it. There is also the **csv2npz.py** script that can be customized to load data from csv text files containing measurements.

* run identify.py, at least supplying the measurement data file and the corresponding
  kinematic model in a .urdf file with some physically consistent CAD parameters as starting point
  (parameters are not necessary to be consistent for all methods but recommended).
Optionally you can supply an output .urdf file path to which the input urdf with exchanged
identified parameters is written. Another measurements file can be supplied for validation. How to call each script is explained when calling the scripts with --help. Options for each task are contained in the .yaml file.


SDP optimization code is based on or uses parts from [cdsousa/wam7\_dyn\_ident](https://github.com/cdsousa/wam7_dyn_ident)

Usage is licensed under the LGPL 3.0, see License.md. Please quote name and authors if you're using this software in any project.
