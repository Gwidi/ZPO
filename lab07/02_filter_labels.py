from glob import glob
import os
import numpy as np

### Desired classes ###

# names:
#   0: person
#   1: bicycle
#   2: car
#   3: motorcycle
#   4: bus
#   5: truck

### Original classes ###

# names:
#  0: person
#  1: bicycle
#  2: car
#  3: motorcycle
#  #
#  5: bus
#  #
#  7: truck
# ... to 79


# update the path if needed
PATH_TO_LABELS = '../data/coco_val2017/labels/'


def filter_file(file_path: str):
    """This function filters the labels of a single file.
    
    Parameters
    ----------
    file_path : str
        The path to the file to filter.
    """
    
    with open(file_path, 'r') as file:
        lines = file.read().split('\n')
        lines = [line.split(' ') for line in lines if line]
    
    new_lines = []
    
    for line in lines:
        # example line: ['0', '0.123', '0.456', '0.789', '0.012']
        
        # TODO: implement the filtering
        # 1. read the class_id as variable
        # 2. cast the class_id to integer
        # 3. filter the class_id
        # 3.1 - remap the class_id to the new class_id if needed
        # 3.2 - if the class_id is not in the desired classes, skip the line by using the `continue` statement
        # 4. overwrite the class_id in the line
        
        class_id = int(line[0])
        if class_id >= 7:
            continue
        else:
            if class_id < 4:
                line[0] = str(class_id)
            elif  class_id == 5:
                class_id = 4
            elif class_id == 7:
                class_id = 5
            elif class_id == 4 or class_id == 6:
                continue
        
        
        
        ### TODO: END ###
        new_lines.append(line)
        
    with open(file_path, 'w') as file:
        for line in new_lines:
            file.write(' '.join(line) + '\n')

if __name__ == '__main__':
    for file_path in sorted(glob(os.path.join(PATH_TO_LABELS, '*.txt'))):
        filter_file(file_path)