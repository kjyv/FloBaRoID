#!/bin/sh

#./identification.py --model ../model/bigman_left_arm.urdf --measurements ../data/LARM/SIM/measurements.npz --plot --explain
#./identification.py --model ../model/kuka_lwr4.urdf --measurements ../data/KUKA/HW/measurements_{1,3}  --validation ../data/KUKA/HW/measurements_2.npz --plot --explain

./identification.py --model ../model/kuka_lwr4_noisy_0.05.urdf --model_real ../model/kuka_lwr4.urdf --measurements ../data/KUKA/SIM/measurements_1.npz --validation ../data/KUKA/SIM/measurements_2.npz --plot --explain
