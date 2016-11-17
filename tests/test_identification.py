
import subprocess
import os

def test_identification():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    output = subprocess.check_output('./examples/identify_kuka_lwr4.sh', cwd=path, shell=True)
    print(output)

if __name__ == '__main__':
    test_identification()
