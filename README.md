# dynamical-system-identification

* excite a robot with fourier series trajectories and record state and torque measurements (using [Yarp](https://github.com/robotology/yarp) or ROS/MoveIt! at the moment)
* identify mass, com, and inertia parameters using a URDF model description of the robot and from the measurements
* at the same time, constrain parameters to physical consistent standard solution space, regardless if input data is well-conditioned or not

details:
* find ideal excitation trajectories with non-linear optimization (as parameterized fourier-series to get periodic trajectories) 
* velocity and acceleration values are derived from position readings, both are zero-phase low-pass filtered
* from supplied measurements, it is optionally possible to only select a percentage of well-conditioned data blocks to decrease the overall condition number of the input data
* implemented estimation methods:
  * weighted least squares instead of ordinary least squares (Zak)
  * estimation of parameter error in addition to absolute parameters using previously known CAD values (Gautier)
  * determine essential parameters (Gautier), estimating only those that are the most certain for the measurement data and leaving the others unchanged
  * formulating identification and constraints as linear SDP optimization problem to get optimal physical consistent parameters (Sousa)
* verification with other measurement data
* save identified values back to URDF file

requirements for identification:
* python 2.7
* python modules: numpy, scipy, sympy, matplotlib, iDynTree, pyyaml, colorama, humanize,
  pylmi-sdp, cvxopt, pyOpt (apply included patch if getting inf's while optimizing trajectories)
* optionally for output as html: mpld3, jinja2

requirements for excitation:
* for yarp/walkman: c compiler, installed [robotology-superbuild](https://github.com/robotology-playground/robotology-superbuild) module, python modules: yarp
* for ros/kuka: [kuka-lwr package](https://github.com/CentroEPiaggio/kuka-lwr), python modules: ros, moveit_msg, moveit_commander
* for other robots, new modules might have to be written

known issues:
* excitation modules are not really generic
* using position control over yarp is suboptimal and can expose timing issues (seems especially with
  additional python to c bridge)
* COM constraints need stl mesh files for the model

usage:

* copy one of the existing .yaml configuration files and customize for your setup

* get joint torque measurements from your robotic system

   possible e.g. by using the excite.py script which filters the measurements as well as getting velocity and acceleration from position measurements.
   If using some other means of movement and data recording, the data files of the numpy data files need to have the expected data fields (see source) and data needs to be filtered

* run identification.py, at least supplying the measurement data file and the corresponding kinematic model in a .urdf file with some physically consistent CAD parameters

   optionally supply an output .urdf file path (which will be the input model with the identified
   parameters), a separate measurement file for validation
   (options are in mostly in .yaml file, others can be seens when calling with -h)

