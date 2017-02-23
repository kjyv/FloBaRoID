
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
        q[0] = t0 * t3 * t4 - t1 * t2 * t5  #x
        q[1] = t0 * t2 * t5 + t1 * t3 * t4  #y
        q[2] = t1 * t2 * t4 - t0 * t3 * t5  #z
        q[3] = t0 * t2 * t4 + t1 * t3 * t5  #w
        return q

    @classmethod
    def fromSO3(self, rotMat):
        """ Return quaternion from rotation matrix. """

        """
        # transforms3d
        M = np.array(rotMat, dtype=np.float64, copy=False)[:4, :4]
        q = np.empty((4, ))
        t = np.trace(M)
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + 1
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
        q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * 1)
        if q[0] < 0.0:
            np.negative(q, q)
        return q
        """

        '''
        # siciliano
        def sign(x):
            if x < 0:
                return -1
            else:
                return 1
        r = rotMat
        x = 0.5 * (sign(r[2,1]-r[1,2]) * np.sqrt(r[0,0]-r[1,1]-r[2,2] + 1) )
        y = 0.5 * (sign(r[0,2]-r[2,0]) * np.sqrt(r[1,1]-r[2,2]-r[0,0] + 1) )
        z = 0.5 * (sign(r[1,0]-r[0,1]) * np.sqrt(r[2,2]-r[0,0]-r[1,1] + 1) )

        w = 0.5 * np.sqrt(np.sum(np.diag(rotMat)) + 1)

        return [x,y,z,w]
        '''

        # from "Converting a Rotation Matrix to a Quaternion" by Mike Day
        r = rotMat

        if r[2,2] < 0:
            if r[0,0] > r[1,1]:
                t = 1 + r[0,0] - r[1,1] - r[2,2]
                q = np.array([t, r[0,1]+r[1,0], r[2,0]+r[0,2], r[1,2]-r[2,1]], dtype=np.float64)
            else:
                t = 1 - r[0,0] + r[1,1] - r[2,2]
                q = np.array([r[0,1]+r[1,0], t, r[1,2]+r[2,1], r[2,0]-r[0,2]], dtype=np.float64)
        else:
            if r[0,0] < -r[1,1]:
                t = 1 - r[0,0] - r[1,1] + r[2,2]
                q = np.array([r[2,0]+r[0,2], r[1,2]+r[2,1], t, r[0,1]-r[1,0]], dtype=np.float64)
            else:
                t = 1 + r[0,0] + r[1,1] + r[2,2]
                q = np.array([r[1,2]-r[2,1], r[2,0]-r[0,2], r[0,1]-r[1,0], t], dtype=np.float64)
        q *= 0.5 / np.sqrt(t)

        return q

    @classmethod
    def toSO3(self, quaternion):
        """return rotation matrix for quaternion q"""

        """
        q = np.array(quaternion, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        if n < np.finfo(float).eps * 4.0:  #test if close to zero
            return np.identity(3)
        q *= np.sqrt(2.0 / n)
        q = np.outer(q, q)

        return np.array([
            [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
            [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
            [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]],
           ])
        """

        x, y, z, w = quaternion
        xx = x ** 2
        yy = y ** 2
        zz = z ** 2
        ww = w ** 2
        xy = x * y
        wz = w * z
        xz = x * z
        wy = w * y
        yz = y * z
        wx = w * x

        return np.array([
            [2*(ww + xx) - 1,  2*(xy - wz),         2*(xz + wy)],
            [2*(xy + wz),      2*(ww + yy) - 1,     2*(yz - wx)],
            [2*(xz - wy),      2*(yz + wx),     2*(ww + zz) - 1]
            ])
