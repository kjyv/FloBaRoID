
Generate trajectories / excitations for Walk-Man

Includes a yarp GYM module that has to be built and started:

in robotCommunication/yarpGYM/:

mkdir build && cd build
cmake ../
make

(have 'gazebo' and 'yarpserver --write' running)
./excitation

using $ yarp write ... /excitation/switch:i
>> start

the control thread is started.

Then, using $ yarp write ... /excitation/command:i
it is then possible to manually set arm positions, e.g. by writing
(set_left_arm -50 40 40 -30 20 10 10) 0

To generate excitation trajectories and send them to the robot, run excite.py --yarp [...]

This will also read the resulting joint torques measurements and write them to a file measurements.npy
