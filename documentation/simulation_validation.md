# Simulation Validation: KUKA LWR4+ Real vs Simulated Measurements

Comparison of real measurements (data/KUKA/HW/measurements_1.npz, recorded 2017 at IIT Genova)
against ideal inverse dynamics computed from the same recorded positions using the URDF model
with friction (Fv*vel + Fc*sign(vel)).

## Torque Residuals (real - simulated)

| Joint | Residual RMS (Nm) | Real RMS (Nm) | Relative | Notes |
|-------|-------------------|---------------|----------|-------|
| 0 | 0.36 | 0.99 | 36% | Large bias (+0.85 Nm) |
| 1 | 1.55 | 25.20 | 6% | Good match, small bias |
| 2 | 0.49 | 3.05 | 16% | Reasonable |
| 3 | 0.98 | 10.67 | 9% | Good match |
| 4 | 0.63 | 0.81 | 77% | Large relative error, bias -0.72 Nm |
| 5 | 0.49 | 0.28 | 173% | Model overpredicts (Fv too high?) |
| 6 | 0.27 | 0.25 | 105% | Model overpredicts |

## Observations

1. **Proximal joints (1-3)**: rigid body model + URDF friction fits well (6-16% residual).
   The URDF inertial parameters are a good match for the IIT robot.

2. **Distal joints (4-6)**: large relative residuals. The small torque signals mean that
   unmodeled effects (stiction, cable forces, sensor bias) dominate over the rigid body torques.
   Joint 5 model overpredicts — likely the URDF Fv=0.3 is too high for this joint.

3. **Mean biases (0.1-0.9 Nm per joint)**: constant offsets that Coulomb friction identification
   should capture. The URDF Fc values (0.05-0.8 Nm) are in the right ballpark.

4. **Real data has zero high-frequency content** (< 0.0001 Nm above 10 Hz): the recorded data
   was heavily filtered by FloBaRoID's preprocessing (filterLowPass1: [8.0, 5] = 8 Hz 5th-order
   Butterworth). The simulated raw sensor noise (0.5% at 600 Hz) is realistic for the raw signal
   but gets removed by preprocessing — this is correct behavior.

## Implications for Simulator

- The simulator's noise model is appropriate for raw sensor data. The identification pipeline's
  own filtering will clean it up, matching what happens with real data.
- The URDF friction values (Fv, Fc) are reasonable starting points but will differ from the
  real robot (wear, individual unit variation). This is expected — identification is meant
  to find the actual values.
- The main unmodeled effects visible in the residuals are: friction model mismatch on distal
  joints, sensor bias/offset, and possibly cable routing forces.
