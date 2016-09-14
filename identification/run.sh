#!/bin/sh

./identification.py --model ../model/kuka_lwr4_cogimon.urdf --model_real ../model/kuka_lwr4.urdf --measurements ../data/KUKA/SIM/measurements_1.npz --validation ../data/KUKA/SIM/measurements_2.npz
