
import subprocess
import os

def test_identification():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    try:
        output = subprocess.check_output('./examples/identify_kuka_lwr4.sh', cwd=path, shell=True)
    except subprocess.CalledProcessError as e:
        print (e.output)
        return (e.returncode)

if __name__ == '__main__':
    test_identification()
