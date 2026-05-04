#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import requests
import logging

logger = logging.getLogger(__name__)

def post(url, data=None, token=""):
    '''Request a POST'''
    try:
        headers = {
            'Authorization': f'Bearer {token}'
        }

        if data is not None:
            response = requests.post(url, headers=headers, json=data)
        else:
            response = requests.post(url, headers=headers)

        # if response.status_code != 200:
        #     logger.error(f"Server returned an error: {response.status_code} {response.json()}")
        
        #access_token = response.json().get('access_token')
        return response.status_code, response.json()
    except Exception as ex:
        logger.error(f"POST request failed - {ex}")
        return 500, {}


def get(url, token=""):
    '''Request a GET with JWT token(Process json)'''
    status, response = get_raw(url, token)
    return status, response.json()


def get_raw(url, token=""):
    '''Request a GET with JWT token(Raw response)'''
    try:
        # Request a GET with JWT token
        headers = {
            'Authorization': f'Bearer {token}'
        }
        response = requests.get(url=url, headers=headers)

        # if response.status_code != 200:
        #     logger.error(f"Server returned an error: {response.status_code}")

        return response.status_code, response
    except Exception as ex:
        logger.error(f"GET request failed - {ex}")
        return None, None