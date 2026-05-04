#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import datetime
import config

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scenario import Scenario

import logging

logger = logging.getLogger(__name__)

def format_date(date):
    return date.strftime("%Y-%m-%d %H:%M:%S.%f")

class TrainRecorder():
    def __init__(self, scenario_handler: "Scenario"):
        self.scenario = scenario_handler
        self.scenario_name = scenario_handler.name
        self.train_date = datetime.datetime.fromtimestamp(self.scenario.scenario_start_time)

        # Result variable(Contains entire records)
        self.step_records = []
        # Temp variable for store
        self.record_temp = {'actions': []}

    def newStepRecord(self, step_number, step_id, step_name):
        '''Make a new record.'''
        self.record_temp['step_number'] = step_number
        self.record_temp['step_name'] = f"{step_id} - {step_name}"
        self.record_temp['step_start_date'] = format_date(datetime.datetime.now())

    def recordAction(self, action):
        '''Record an action to step record'''
        result = {}
        result['timestamp'] = format_date(datetime.datetime.now())
        result['action'] = action
        self.record_temp['actions'].append(result)
        logger.info(f"Recorded action: {action}")

    def saveStepRecord(self):
        '''Save a record and cleanup temp var'''
        # Get elapsed time
        _, self.record_temp['elapsed_time'] = self.scenario.getRunningTimes()

        self.step_records.append(self.record_temp)
        self.record_temp = {'actions': []}

    def finalizeRecord(self, filename_recorded=None):
        '''Finalize the record and return the result'''
        result = {}
        result['trainee_idx'] = None
        result['user_id'] = "local"
        result['scenario_name'] = self.scenario_name
        result['start_date'] = format_date(self.train_date)
        result['end_date'] = format_date(datetime.datetime.now())
        result['step_records'] = self.step_records
        if config.record_video:
            result['train_video'] = filename_recorded
        else:
            result['train_video'] = ""

        return result