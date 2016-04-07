
# Excitation scripts for generating trajectories and recording measurements

## Generate trajectories / excitations for Walk-Man

A Yarp GYM module is included that has to be built and started:

in robotCommunication/yarpGYM/:

mkdir build && cd build
cmake ../
make

(have 'gazebo' and 'yarpserver --write' running)
./excitation

$ yarp write ... /excitation/switch:i
>> start

The control thread is now started and accepts commands.

Using $ yarp write ... /excitation/command:i
it is then possible to manually set arm positions, e.g. by writing
(set_left_arm -50 40 40 -30 20 10 10) 0

To generate excitation trajectories and send them to the robot, run excite.py --yarp [...]
This will also read the resulting joint torques measurements and write them to a file measurements.npy

## Generate excitation for Kuka LWR4+ (using ROS package from https://github.com/CentroEPiaggio/kuka-lwr)

start controllers, simulator and moveit (to directly use the hardware add options: use_lwr_sim:=false lwr_powered:=true)
$ roslaunch single_lwr_launch single_lwr.launch load_moveit:=true

(make sure that gazebo plugin gets loaded in world file and joint_state_publisher has high enough rate param of 100-200 Hz set in launch file)

Call excitation script using its ros module:
$ ./excite.py --ros [...]
