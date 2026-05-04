#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import os
import json

def handleImport(data):
    '''Check objects loaded from json and process import macros.
       This method is run recursively.'''
    # Handle list
    if isinstance(data, list):
        result = []
        for subdata in data:
            # Handle import
            if isinstance(subdata, str) and subdata.startswith("import"):
                import_path = subdata.split(":")[1]
                # Open import file
                with open(import_path, encoding='utf8') as f:
                    import_load = json.load(f)

                # Append imported objects
                for import_obj in import_load:
                    result.append(handleImport(import_obj))

            else:
                result.append(handleImport(subdata))
    
    # Handle dict
    elif isinstance(data, dict):
        result = {}
        for key, subdata in data.items():
            result[key] = handleImport(subdata)
    
    # Handle others
    else:
        return data
    
    return result

def readScenarioFile(file_path):
    '''Read a scenario json file with handling imports.'''

    # Open a scenario json file
    with open(file_path, encoding='utf8') as f:
        scenario_load = json.load(f)
    
    result = handleImport(scenario_load)
    
    return result


if __name__ == '__main__':
    print(readScenarioFile("scenarios/Helmet/scenario.json"))