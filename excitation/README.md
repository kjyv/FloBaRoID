
# Excitation scripts for generating trajectories and recording measurements

## Trajectory optimization

Using global non-linear optimization (particle swarm optimization), parameters for trajectories can be found that minimize the condition number of the regressor retrieved through the generated trajectory (in simulation of course). Set the option `optimizeTrajectory` in your .yaml file. It is important to set good bounds to narrow the search space and to have proper joint angle, velocity and torque limits in the URDF model file. See further comments in the sample configuration files.


## Excitation modules

Existing modules that will send commands to a robot and record measurements can be used with slight modifications or as template for other robot command interfaces.

## Generate excitation for Walk-Man

A Yarp GYM module is included that has to be built and started:

in robotCommunication/yarpGYM/:

`mkdir build && cd build`   
`cmake ../`   
`make`

(have 'gazebo' and 'yarpserver --write' running)
`./excitation`

`$ yarp write ... /excitation/switch:i`   
`>> start`

The control thread is now started and accepts commands.

Using $ yarp write ... /excitation/command:i
it is then possible to manually set arm positions, e.g. by writing
`(set_left_arm -50 40 40 -30 20 10 10) 0`

To generate excitation trajectories and send them to the robot, set 
the option exciteMethod to 'yarp' and run `./excite.py [...]`.
This will also read the resulting joint torques measurements and write them to a file measurements.npy

## Generate excitation for Kuka LWR4+

(using ROS package from https://github.com/CentroEPiaggio/kuka-lwr)

start controllers, simulator and moveit (to directly use the hardware add options: use\_lwr\_sim:=false lwr\_powered:=true)
`$ roslaunch single_lwr_launch single_lwr.launch load_moveit:=true`

(make sure that gazebo plugin gets loaded in world file and joint\_state\_publisher has high enough rate param of 100-200 Hz set in launch file)

To generate excitation trajectories and send them to the robot, set 
the option exciteMethod to 'ros' and run `./excite.py [...]`.


## Measurements data file structure

The measurements retrieved from excitation are saved in a numpy .npz binary file archive which includes multiple data streams. All
data fields have the same amount of samples S relative to the time in field 'times'. The same structure is expected by identification.py.

|field name|content|
|---|---|
|positions | joint positions in radians, SxN_DOF|
|positions_raw | unfiltered joint positions, SxN_DOF|
|velocities | joint angular velocity in rad/sec, SxN_DOF|
|velocities_raw | unfiltered joint angular velocities in rad/sec, SxN_DOF|
|accelerations | joint angular accelerations, SxN_DOF|
|torques | measured torques of each joint, SxN_DOF|
|torques_raw | unfiltered torques of each joint, SxN_DOF|
|base_velocity | linear (0-2) and angular (3-5) velocity of the base link, Sx6|
|base_acceleration |  linear (0-2) and angular (3-5) acceleration of the base link, Sx6|
|contacts | measured external contact wrenches, dictionary {'urdf frame name': Sx6}|
|times | time of each sample in sec, Sx1|
|frequency | frequency of measured values in Hz, 1 value|
