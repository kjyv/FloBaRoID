#!/bin/sh

#./identification.py --model ../model/bigman_left_arm.urdf --measurements ../data/LARM/SIM/measurements.npz --plot --explain
#./identification.py --model ../model/kuka_lwr4.urdf --measurements ../data/KUKA/HW/measurements_{1,3}  --verification ../data/KUKA/HW/measurements_2.npz --plot --explain

./identification.py --model ../model/kuka_lwr4_noisy.urdf --model_output ../model/kuka_lwr4.urdf --measurements ../data/KUKA/SIM/measurements_{1,3}.npz  --verification ../data/KUKA/SIM/measurements_2.npz --plot --explain
