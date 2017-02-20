
import math
import numpy as np

class Quaternion(object):

    @classmethod
    def fromRPY(self, roll, pitch, yaw):
        t0 = np.cos(yaw * 0.5)
        t1 = np.sin(yaw * 0.5)
        t2 = np.cos(roll * 0.5)
        t3 = np.sin(roll * 0.5)
        t4 = np.cos(pitch * 0.5)
        t5 = np.sin(pitch * 0.5)

        q = np.zeros(4)
        q[0] = t0 * t2 * t4 + t1 * t3 * t5
        q[1] = t0 * t3 * t4 - t1 * t2 * t5
        q[2] = t0 * t2 * t5 + t1 * t3 * t4
        q[3] = t1 * t2 * t4 - t0 * t3 * t5
        return q

    @classmethod
    def fromSO3(self, rotMat, isprecise=False):
        """Return quaternion from rotation matrix.

        If isprecise is True, the input matrix is assumed to be a precise rotation
        matrix and a faster algorithm is used.
        """

        M = np.array(rotMat, dtype=np.float64, copy=False)[:4, :4]
        if isprecise:
            q = np.empty((4, ))
            t = np.trace(M)
            if t > M[3, 3]:
                q[0] = t
                q[3] = M[1, 0] - M[0, 1]
                q[2] = M[0, 2] - M[2, 0]
                q[1] = M[2, 1] - M[1, 2]
            else:
                i, j, k = 0, 1, 2
                if M[1, 1] > M[0, 0]:
                    i, j, k = 1, 2, 0
                if M[2, 2] > M[i, i]:
                    i, j, k = 2, 0, 1
                t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
                q[i] = t
                q[j] = M[i, j] + M[j, i]
                q[k] = M[k, i] + M[i, k]
                q[3] = M[k, j] - M[j, k]
                q = q[[3, 0, 1, 2]]
            q *= 0.5 / math.sqrt(t * M[3, 3])
        else:
            m00 = M[0, 0]
            m01 = M[0, 1]
            m02 = M[0, 2]
            m10 = M[1, 0]
            m11 = M[1, 1]
            m12 = M[1, 2]
            m20 = M[2, 0]
            m21 = M[2, 1]
            m22 = M[2, 2]
            # symmetric matrix K
            K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                             [m01+m10,     m11-m00-m22, 0.0,         0.0],
                             [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                             [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
            K /= 3.0
            # quaternion is eigenvector of K that corresponds to largest eigenvalue
            w, V = np.linalg.eigh(K)
            q = V[[3, 0, 1, 2], np.argmax(w)]
        if q[0] < 0.0:
            np.negative(q, q)
        return q

    @classmethod
    def toSO3(self, quaternion):
        """return rotation matrix for quaternion q"""

        q = np.array(quaternion, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        if n < np.finfo(float).eps * 4.0:  #test if close to zero
            return np.identity(4)
        q *= math.sqrt(2.0 / n)
        q = np.outer(q, q)

        return np.array([
            [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
            [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
            [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]],
            ])
