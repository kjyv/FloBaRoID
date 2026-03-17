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

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync
```

### System dependencies

* **suite-sparse** (required for building cvxopt): `brew install suite-sparse` (macOS) or `apt install libsuitesparse-dev` (Ubuntu/Debian)
* **eigen3, swig** (required for building iDynTree): `brew install eigen@3 swig` (macOS) or `apt install libeigen3-dev swig` (Ubuntu/Debian)
* **ipopt** (for iDynTree build): `brew install ipopt` (macOS) or `apt install coinor-libipopt-dev` (Ubuntu/Debian)
* **dsdp5** (command line executable, required for SDP-constrained identification)

### Optional extras

```bash
uv sync --extra tikz-plots      # matplotlib2tikz
uv sync --extra parallel         # mpi4py
```

### Additional non-PyPI dependencies

* symengine.py (to speedup SDP, optional)
* rbdl (alternative for inverse dynamics, optional)

requirements for excitation module:

* for ros, python modules: ros, moveit\_msg, moveit\_commander
* for yarp: c compiler, installed [robotology-superbuild](https://github.com/robotology-playground/robotology-superbuild), python modules: yarp
* for other robots, new modules will have to be written

requirements for optimization module:

* optimization: python modules: iDynTree, pyOpt (fork at https://github.com/kjyv/pyOpt is recommended)
* pyipopt from https://github.com/xuy/pyipopt (plus cmd line ipopt/libipopt with libhsl/coin-hsl)
* mpi4py / mpirun (for parallel trajectory optimization)
* [fcl 0.5.0](https://github.com/flexible-collision-library/fcl/releases) and python-fcl (from https://github.com/jf---/python-fcl) (possibly disable octomap if there are errors)

Also see the [Tutorial](documentation/TUTORIAL.md).

Known limitations:

* trajectory optimization is limited to fixed-base robots (full simulation, balance criterion etc. not implemented)
* YARP excitation module is not very generic (ROS should be)
* using position control over YARP is not realtime safe and can expose timing issues (especially with python to C bridge)
* Since preparing SDP matrices uses sympy expressions, most of the time for solving the identification problem is spent in symbolic manipulations rather than the actual convex optimization solver. Possibly the time demands can be reduced.

SDP optimization code is based on or uses parts from [cdsousa/wam7\_dyn\_ident](https://github.com/cdsousa/wam7_dyn_ident)

Usage is licensed under the LGPL 3.0, see License.md. Please quote the following publication if you're using this software for any project:
`S. Bethge, J. Malzahn, N. Tsagarakis, D. Caldwell: "FloBaRoID — A Software Package for the Identification of Robot Dynamics Parameters", 26th International Conference on Robotics in Alpe-Adria-Danube Region (RAAD), 2017`
