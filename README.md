# dynamical-system-identification

* excite a robot with fourier series trajectories and record state and torque measurements (using [Yarp](https://github.com/robotology/yarp) or ROS/MoveIt! at the moment)
* identify mass, com, and inertia parameters using a URDF model description of the robot and from the measurements
* at the same time, constrain parameters to physical consistent standard solution space, regardless if input data is well-conditioned or not

details:
* optimzied excitation trajectories (parameterized fourier-series to get periodic trajectories) 
* acceleration and velocity values are derived from position readings, both are zero-phase low-pass filtered
* from supplied measurements, it is optionally possible to only select a percentage of well-conditioned data blocks to decrease the overall condition number of the input data
* implemented estimation methods:
  * weighted least squares instead of ordinary least squares (Zak)
  * estimation of parameter error in addition to absolute parameters using previously known CAD values (Gautier)
  * determine essential parameters (Gautier), estimating only those that are most relevant for the measurement data and leaving the others unchanged
  * formulating identification and constraints as linear SDP optimization problem (Sousa)
* verification with other measurement data
* save identified values back to URDF file

requirements for identification:
* python 2.7
* python modules: numpy, scipy, sympy, matplotlib, iDynTree, pyyaml, colorama, humanize,
  pylmi-sdp, cvxopt, pyOpt, (optional: mpi4py < 2.0.0)
* optionally for output as html: mpld3, jinja2

requirements for excitation:
* for yarp/walkman: c compiler, installed [robotology-superbuild](https://github.com/robotology-playground/robotology-superbuild) module, python modules: yarp
* for ros/kuka: [kuka-lwr package](https://github.com/CentroEPiaggio/kuka-lwr), python modules: ros, moveit_msg, moveit_commander
* for other robots, new modules might have to be written

known issues:
* excitation methods are not really generic 
* using position control over yarp is suboptimal and can expose timing issues
* ros timing might also lead to missed data packages

usage:

* get joint torque measurements from a robotic link structure

   possible e.g. by using the excite.py script which filters the measurements as well (containts no generic trajectory generation yet)
   if using some other means of movement and data recording, the data files of the numpy data files need to have the expected data fields and data needs to be filtered

* run identification.py, at least supplying the measurement data file and the corresponding kinematic model .urdf file with some physically consistent CAD parameters

   optionally an output .urdf file path, separate measurements for validation, some display options
   possibly it's necessary to set some options for identification methods in the beginning of identification.py

