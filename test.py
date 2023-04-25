

from matplotlib import pyplot as plt
import cv2
import numpy as np
import subprocess
result = subprocess.run(['python3', 'classify.py', '--data=Results',
                         '--model=classifier.model'], capture_output=True)
result = result.stdout.decode()
split_result = result.split('\n')

for i in split_result:

    if int(i[2:4]) <= 20:
        if 'healthy' not in i:
            print(i)
    if int(i[2:4]) > 20:
        if 'sick' not in i:
            print(i)
    if 'im40' in i:
        break
print(split_result[-2])
