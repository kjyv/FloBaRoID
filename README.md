# FloBaRoID [![Build Status](https://travis-ci.org/kjyv/FloBaRoID.svg?branch=master)](https://travis-ci.org/kjyv/FloBaRoID)

(FLOating BAse RObot dynamical IDentification)

FloBaRoID is a python toolkit for parameter identification of floating-base rigid body tree-structures such as
humanoid robots. It aims to provide a complete solution for obtaining physical consistent identified dynamics parameters.

<div>
<img alt="Overview diagram" src="https://cdn.rawgit.com/kjyv/FloBaRoID/master/documentation/identification_overview.svg" width="57%" align="left" hspace="5px">
<img alt="Visualization of Kuka LWR4+" src="documentation/kuka_vis.png" width="38%">
</div>

Tools:

* **trajectory.py**: generate optimized trajectories
* **excite.py**: send trajectory to control the robot movement and record the resulting measurements
* **identify.py**: identify dynamical parameters (mass, COM and rotational inertia) starting from an URDF description and from torque and force measurements
* **visualize.py**: show 3D robot model of URDF, trajectory motion


Features:

* find optimized excitation trajectories with non-linear global optimization (as parameters of fourier-series for periodic soft trajectories) 
* data preprocessing
    * derive velocity and acceleration values from position readings
    * data is zero-phase low-pass filtered from supplied measurements
    * it is possible to only select a combination of data blocks to yield a better condition number (Venture, 2009)
* validation with other measurement files
* excitation of robots, using ROS/MoveIt! or Yarp
* implemented estimation methods:
  * ordinary least squares, OLS
  * weighted least squares (Zak, 1994)
  * estimation of parameter error using previously known CAD values (Gautier, 2013)
  * essential standard parameters (Pham, Gautier, 2013), estimating only those that are most certain for the measurement data and leaving the others unchanged
  * identification problem formulation with constraints as linear convex SDP problem to get optimal physical consistent standard parameters (Sousa, 2014)
  * non-linear optimization within consistent parameter space (Traversaro, 2016)
* visualization of trajectories
* plotting of measured and estimated joint state and torques (interactive, HTML, PDF or Tikz)
* output of the identified parameters directly into URDF

requirements for identification module:

* python 2.7 or >=3.3
* python modules
	* numpy (> 1.8), scipy, sympy (>= 1.0), pyyaml, numpy-stl, cvxopt, pylmi-sdp, matplotlib (>= 1.4), colorama, palettable, humanize, tqdm
	* iDynTree, e.g. from [iDynTree superbuild](https://github.com/robotology/idyntree-superbuild/) (with enabled python binding)
	* when using Python 2.7: future
	* when using Python <3.5: typing

(You can do `pip install -r requirements.txt` for most of them)

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
* pyipopt from https://github.com/xuy/pyipopt (plus cmd line ipopt/libipopt with libhsl)
* mpi4py / mpirun (for parallel trajectory optimization)


Quick start tutorial:

The goal of identification is to find dynamic model parameters from measurements of motions. The options for all steps of this task are held within a configuration file in the config/ dir.

1. copy one of the existing .yaml configuration files and customize 
it for your setup.
`cp config/kuka_lwr.yaml config/example.yaml`

2. Use the trajectory.py script to generate an optimal exciting trajectory (only fixed base at the moment). Enable the corresponding options in the configuration and optionally supply a world urdf file that includes objects that the robot might collide with, e.g. a table. This might take a while depending on the degrees of freedom, prefix with `mpirun` to parallelize this. The output file will contain the found parameters of the trajectory.
`./trajectory.py --config configs/example.yaml --model model/example.urdf --world model/world.urdf`

3. get joint torque measurements for the trajectory from your robotic system, if suitable by using the excite.py script. It will load the previously created trajectory file and move the robot through a module or alternatively simulate the torques using the supplied model. If necessary, look at the existing modules and write a custom one. After retrieving measurements, filtering as well as deriving velocity and acceleration is done. If you are using other means of motion control and data recording, filter the data and write the data into numpy container files that have the expected data fields (see README.md in ./excitation/). There is also the **csv2npz.py** script that loads data from csv text files containing raw measurements, preprocesses them and writes to the container format (customize it for your columns etc.).
`./excite.py --model model/example.urdf --config configs/example.yaml --plot \`
`--trajectory model/example.urdf.trajectory.npz --filename measurements.npz`

4. Finally, run identify.py on the measurements and the corresponding
  kinematic model in a .urdf file with some CAD parameters as starting point (parameters don't have to be consistent but recommended). Optionally you can supply an output .urdf file path to which the input urdf with exchanged
identified parameters is written. Another measurements file can be supplied for validation.
`./identify.py --config configs/example.yaml  --model model/example.urdf --measurements \`
`measurements.npz --verify measurements_2.npz --output model/example_identified.urdf`

known issues:

* trajectory optimization is limited to fixed-base robots ATM
* trajectory optimization does not yet check for self or other collisions (e.g. objects from a world URDF)
* YARP excitation module is not generic (ROS should be)
* using position control over yarp is suboptimal and can expose timing issues (seems to happen especially with used python to c bridge)
* COM constraints need an stl mesh files for the model to compute the enclosing hull, doesn't e.g. read geometric shape definitions for link

SDP optimization code is based on or uses parts from [cdsousa/wam7\_dyn\_ident](https://github.com/cdsousa/wam7_dyn_ident)

Usage is licensed under the LGPL 3.0, see License.md. Please quote name and authors if you're using this software in any project.
