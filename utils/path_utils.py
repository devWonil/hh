#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import os

class PathUtil:
    def __init__(self, scenario_name):
        self.scenario_name = scenario_name

    def getPath(self, file):
        '''First, finds file from scenario folder.
           If exists, returns file path of scenario folder.
           If not exists, finds file from global folder and returns.'''
        
        local_path = f"scenarios/{self.scenario_name}/assets/"
        global_path = "assets/"
        if os.path.isfile(local_path + file):
            return local_path + file
        elif os.path.isfile(global_path + file):
            return global_path + file
        else:
            raise FileNotFoundError(f"File {file} not found in both {local_path} and {global_path}.")