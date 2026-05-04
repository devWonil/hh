#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import sys
import json

'''
This script processes a JSON log file containing detection data, extracts relevant information, and saves it in a new JSON file format.
Usage: python process_log.py <filename>
'''

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_log.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # print(json.dumps(data, indent=4, ensure_ascii=False))
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{filename}' is not a valid JSON file.")

    return_txt = "["

    for detect in data['detects']:
        detect_data = {}
        detect_data['time'] = detect['time']
        detect_data['step'] = detect['step']
        detect_data['clicking_button_info'] = detect['clicking_button_info']
        detect_data['clicking_button_name'] = detect['clicking_button_name']
        detect_data['clicking_button_state'] = detect['clicking_button_state']

        # Serialize mediapipe detect datas
        detect_json_txt = '\n  ' + json.dumps(detect_data) + ',' 
        return_txt += detect_json_txt
            
    processed_filename = f"{filename.rsplit('.', 1)[0]}_processed.json"
    with open(processed_filename, 'w', encoding='utf-8') as processed_file:
        processed_file.write(return_txt[:-1] + '\n]')
    print(f"Processed data saved to '{processed_filename}'.")

if __name__ == "__main__":
    main()