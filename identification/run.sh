#!/bin/sh

#./identification.py --model ../model/bigman_left_arm.urdf --measurements ../data/LARM/SIM/measurements.npz --plot --explain
./identification.py --model ../model/kuka_lwr4.urdf --measurements ../data/KUKA/HW/measurements_1.npz --plot --explain
