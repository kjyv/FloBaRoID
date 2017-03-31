
# Excitation scripts for generating trajectories and recording measurements

## Trajectory optimization

Using global non-linear optimization (particle swarm optimization), parameters for trajectories can
be found that minimize the condition number of the regressor retrieved through the generated
trajectory (in simulation of course). Set the option `optimizeTrajectory` in your .yaml file to
enable this. It is important to set good bounds to narrow the search space and to have proper joint
angle, velocity and torque limits in the URDF model file. See further comments in the sample
configuration files.

## Show trajectory data

In order to show curves of an optimized or previously recorded trajectory again, 
`excite.py --plot --dryrun --filename <measurement file>` can be used. Using
`visualizer.py --config <config> --model <model urdf> --trajectory <measurement file>`
a 3D representation of the model and trajectory can be shown.

## Excitation modules

Existing modules that will send commands to a robot and record measurements can be used with slight
modifications or as template for other robot command interfaces.

## Generate excitation for Walk-Man

A Yarp GYM module is included that has to be built and started:

in robotCommunication/yarpGYM/:

`mkdir build && cd build`   
`cmake ../`   
`make`

run 'yarpserver --write' and then 'gazebo', load the robot

run `./excitation`

`$ yarp write ... /excitation/switch:i`   
`>> start`

The control thread is now started and accepts commands.

Using $ yarp write ... /excitation/command:i
it is then possible to manually set positions, e.g. by writing
`(set_legs_refs 0 0 0 0 0 0 0 0 0 0 0 0) 0`


To generate excitation trajectories and send them to the robot, set 
the option exciteMethod to 'yarp' and run `./excite.py [...]`.
This will also read the resulting joint torques measurements and write them to a file
measurements.npz

## Generate excitation for Kuka LWR4+

(using ROS package from https://github.com/CentroEPiaggio/kuka-lwr)

start controllers, simulator and moveit (to directly use the hardware add options: use\_lwr\_sim:=false lwr\_powered:=true)
`$ roslaunch single_lwr_launch single_lwr.launch load_moveit:=true`

(make sure that gazebo plugin gets loaded in world file and joint\_state\_publisher has high enough rate param of 100-200 Hz set in launch file)

To generate excitation trajectories and send them to the robot, set 
the option exciteMethod to 'ros' and run `./excite.py [...]`.


## Measurements data file structure

The measurements retrieved from excitation are saved in a numpy .npz binary file archive which
includes multiple data streams. All data fields have the same amount of samples S relative to the
time in field 'times'. The same structure is expected by identification.py.

|field name|content|
|---|---|
|positions | joint positions in rad, SxN_DOF|
|[positions_raw] | unfiltered joint positions, SxN_DOF (optional)|
|velocities | joint angular velocity in rad/sec, SxN_DOF|
|[velocities_raw] | unfiltered joint angular velocities in rad/sec, SxN_DOF (optional)|
|accelerations | joint angular accelerations in rad/s<sup>2</sup>, SxN_DOF|
|torques | measured torques of each joint in Nm, SxN_DOF|
|[torques_raw] | unfiltered torques of each joint, SxN_DOF (optional)|
|base_velocity* | linear (0-2) and angular (3-5) velocity of the base link expressed in the world reference frame  in m/s and rad/s, Sx6|
|base_acceleration* |  proper linear (0-2) and angular (3-5) acceleration of the base link (without gravity) expressed in the world reference frame in m/s<sup>2</sup> and rad/s<sup>2</sup>, Sx6|
|base_rpy* |  Orientation of the base link in roll-pitch-yaw order expressed relative to the world reference frame in rad Sx3|
|contacts | measured external contact wrench for each sample, array of dictionaries {'urdf frame name': Sx6}|
|times | time of each sample in sec, Sx1|
|frequency | frequency of measured values in Hz, 1 value|

Values in [] are optional.
Values with * only need to be specified when using floating base dynamics.

All data is expected by identify.py to already be cleaned and low-pass filtered and to not
include any big measurement errors. The noise should ideally be gaussian and have zero mean.

The sampling frequency should be sufficiently high (e.g. at least 100 Hz) to get reasonably good position and velocity derivatives.

The amount of samples should also be high enough for it to contain sufficient information about the parameters. It depends on how many parameters are to be identified and on the motion range of the robot. At least 10
times the amount of parameters is possibly a good rule of thumb, more is always better. The higher
the sampling frequency, the less information there is in successive samples, so the number should be
higher at e.g. 1000 Hz. At the same time, more or less redundant samples can be skipped by setting the skipSamples option to speed up identification.
