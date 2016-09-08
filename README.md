# OpenDSI

Open Dynamical System Identification
- Framework for dynamical system identification of floating-base rigid body structures

Modules:

* excitation: control a robot movement along optimized trajectories and record the state and torque measurements (using [Yarp](https://github.com/robotology/yarp) or ROS/MoveIt! at the moment)
* identification: identify dynamical parameters (mass, com and inertia) starting from an URDF model description of the robot and from measurements
	* parameters are constrained to physical consistent standard parameter space, regardless if input data is well-conditioned or not
* write improved parameters into output URDF ready to use for control or simulation


Details:

* finds ideal excitation trajectories with non-linear global optimization (parameters of fourier-series for periodic soft trajectories) 
* data preprocessing
	* derives velocity and acceleration values from position readings, data is zero-phase low-pass filtered
	* from supplied measurements, it is optionally possible to only select the well-conditioned data to decrease the overall condition number of the input data
* constrained convex optimization to find global optimal solution of the parameter identification problem
* cross-validation with other measurement data
* implemented estimation methods:
  * ordinary least squares, OLS
  * weighted least squares (Zak)
  * estimation of parameter error using previously known CAD values (Gautier)
  * essential standard parameters (Pham, Gautier), estimating only those that are most certain for the measurement data and leaving the others unchanged
  * identification problem formulation with constraints as linear convex SDP problem to get optimal physical consistent parameters (Sousa)

requirements for identification module:

* python 2.7
* python modules: numpy, scipy, sympy, iDynTree, pyyaml, transforms3d, numpy-stl, pylmi-sdp, cvxopt (with dsdp5), matplotlib, colorama, palettable, humanize
* optionally for html plots: mpld3, jinja2

requirements for excitation modules:

* optimization: python modules: iDynTree, pyOpt (apply included patch if getting inf's while optimizing trajectories)
* for yarp/walkman: c compiler, installed [robotology-superbuild](https://github.com/robotology-playground/robotology-superbuild), python modules: yarp
* for ros/kuka: [kuka-lwr package](https://github.com/CentroEPiaggio/kuka-lwr), python modules: ros, moveit\_msg, moveit\_commander
* for other robots, new modules might have to be written

known issues:

* excitation modules are not really generic yet
* using position control over yarp is suboptimal and can expose timing issues (seems to happen especially with used python to c bridge)
* COM constraints need stl mesh files for the model to compute the enclosing hull

(mode or less) Quick start :

* copy one of the existing .yaml configuration files and customize for your setup

* get joint torque measurements from your robotic system   
   possible e.g. by using the excite.py script which filters the measurements as well as getting velocity and acceleration from position measurements.
   If using some other means of movement and data recording, the data files of the numpy data files need to have the expected data fields (see source) and data needs to be filtered

* run identification.py, at least supplying the measurement data file and the corresponding kinematic model in a .urdf file with some physically consistent CAD parameters    
   optionally supply an output .urdf file path (which will be the input model with the identified
   parameters), a separate measurement file for validation
   (options are in mostly in .yaml file, others can be seens when calling with -h)

