# Tutorial

The goal of identification is usually to find dynamic model parameters from measurements of motions. Ideally, some previous parameters are available e.g. from a CAD model which also gives the kinematic model. In the following, an example identification is carried out for the Kuka LWR4+ robot. The options for all steps of this task are held within a configuration file in the config/ dir.

1. copy an existing .yaml configuration file and customize it for your setup with a text editor.
`cp config/kuka_lwr.yaml config/example.yaml`

2. Use the trajectory.py script to generate an optimal exciting trajectory (only fixed base at the moment). The corresponding options in the configuration should be set (for the case of the LWR4+ that is done) and optionally supply a world urdf file that includes the ground and objects that the robot might collide with, e.g. a table. The optimization will simulate each trajectory and check for all constraints to be met while minimizing the condition number of the dynamics regressor. This might take a while depending on the degrees of freedom. You can prefix the call with `mpirun -n <n>` to parallelize this. An output file containing the found parameters of the trajectory will be saved.
`./trajectory.py --config configs/example.yaml --model model/example.urdf --world model/world.urdf`

3. Get joint torque measurements for the trajectory from your robotic system, if suitable by using the excite.py script. It will load the previously created trajectory file and move the robot through the specified module (in the config file). Alternatively, simulation can be enabled to simulate the torques using the supplied model parameters. If necessary, look at the existing modules and write a custom one for your communication method. After retrieving the measurements, filtering as well as deriving velocity and acceleration is done and is saved to a measurements file. If you are using other means of motion control and data recording and don't use the excite.py script, the data needs to be filtered and saved to a numpy data file that has the expected data fields (see README.md in excitation/). There is also the **csv2npz.py** script that loads raw data from csv text files, preprocesses them with the same filtering and writes to the container format (you'll need to customize it for the columns in your csv file etc.).
In this example for the LWR4+, we simply simulate the trajectory file to receive a measurements file.
`./excite.py --model model/example.urdf --config configs/example.yaml --plot \`
`--trajectory model/example.urdf.trajectory.npz --filename measurements.npz`

4. Finally, run identify.py with the measurements file and again the
  kinematic model in a .urdf file with the a priori parameters. These parameters don't have to be physical consistent but it's recommended (they should be when they come from a CAD system). The constrained optimization for identification Optionally you can supply an output .urdf file path to which the input urdf with exchanged
identified parameters is written. Another measurements file can be supplied for validation.
`./identify.py --config configs/example.yaml  --model model/example.urdf --measurements \`
`measurements.npz --verify measurements_2.npz --output model/example_identified.urdf`

The output html file in output/ should look similar to [this output](../documentation/example output/output_kuka.html).
After the plots that compare a priori and identified estimated torques with the measurements, there are tables of the identified parameters and different error measures.
The table columns show parameters for A priori (URDF), Identified and the absolute change between them. There also is a percentual difference value (%e) that is given in relation to the magnitude of the a priori value.

Different estimation error measures that are given are
Absolute mean error:
The mean over the error vector norms for each joint.

Relative mean error: 
The absolute mean error normalized with the norm of the measured data vectors.

Normalized root mean square (NRMS) error:
The square root of the mean over the joints of the squared error, normalized by the possible torque range of each joint (as given in the URDF).
