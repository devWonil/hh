#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import json
import time

# Objects
from .objects.object_button import ObjectButton
from .objects.hand_button import HandButton
from .objects.popup import Popup
from .objects.text import Text
from .objects.image import Image
from .objects.lpips import LPIPSObject

import logging

logger = logging.getLogger(__name__)
import utils.coord_utils as coord
import utils.state as STATE

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scenario import Scenario

# A dummy function for button action
def dummy(*args):
    pass

def loadStep(scenario: "Scenario", step):
    '''Change step to provied index. Reset vars and create buttons'''
    # Stop Guide sound
    if scenario.guide_sound is not None:
        scenario.guide_sound.stop()
    scenario._client.stop_tts()

    # If last running state is paused, commit paused time
    if scenario.running_state == STATE.PAUSED:
        scenario.resume()

    scenario.running_state = STATE.CHANGING

    # Validate step
    if step is None or step < 0:
        logger.warning(f"Unknown step {step}, Moving to first step")
        step = 0

    if step >= scenario.total_steps:
        logger.info("Scenario completed!")
        scenario.recordAction("Scenario completed.")
        scenario._client.unloadScenario()
        return True

    # Change step number
    scenario.current_step = step
    # Load step datas
    step_info = scenario.steps[scenario.current_step]

    # Handle Train recorder
    # If target step is not 0(start), Save recent record
    if step != 0:
        scenario.train_recorder.recordAction("Step ended.")
        scenario.train_recorder.saveStepRecord()

    scenario.train_recorder.newStepRecord(step, step_info['id'], step_info['name'])
    scenario.train_recorder.recordAction("Step started.")

    # init vars
    scenario.completion_events = {}
    scenario.delay_start_time = None
    scenario.delay_wait_time = None
    scenario.delay_state = STATE.NOT_STARTED
    scenario.yolo_track_id = None
    scenario.bg_count = 0
    scenario.bg_current_idx = 0
    scenario.step_pause_time = 0

    scenario.guide_sound_last_played_time = time.time()  # Reset guide sound timer=

    # Reset VLM state for new step
    scenario._vlm_last_check_time = time.time()  # Reset timer to avoid immediate trigger
    scenario.vlm_prompt = step_info.get('vlm_prompt', None)  # Load step-specific VLM prompt

    # Update step start time
    scenario.step_start_time = time.time()

    # Set display runtime option
    if 'display_runtime' in step_info:
        scenario.is_display_runtime = step_info['display_runtime']
    else:
        scenario.is_display_runtime = False

    # =============== Create conditions ==============
    if 'completion_events' in step_info:
        for cond in step_info['completion_events']:
            if cond.startswith("delay:"):
                scenario.delay_wait_time = int(cond.split(":")[1])
                scenario.completion_events['delay'] = False
            else:
                scenario.completion_events[cond] = False

    # =========== Create button object(s) ============
    scenario.buttons.clear()
    get_hands = scenario._client.mp_thread.getHands # Load a method only

    # Define a function to add a button
    def addButton(btn):
        ''' Create a button object from configuration. '''
        if 'angle' in btn:
            angle_deg = btn['angle']
        else:
            angle_deg = 0
        # Activate action if exists
        if 'action' in btn:
            cb_arg = None
            action = btn['action']
            if action == 'RESTART':
                callback = scenario.restart
            elif action == 'PREVIOUS':
                callback = scenario.goToPreviousStep
            elif action == 'NEXT':
                callback = scenario.goToNextStep
            elif action == 'RECORD':
                callback = scenario.recordAction
                cb_arg = f"Button Clicked: {btn['name']}"
            else:
                callback = scenario.changeStep
                cb_arg = action # Step number or index to move
            button = HandButton(scenario._client,
                                get_hands,
                                btn['name'],
                                coord.fromWorldToPixelBbox(btn['bbox']),
                                angle_deg,
                                btn['timer'],
                                callback,
                                cb_arg)

        # completion_event: Trigger step's condition
        elif 'completion_event' in btn:
            button = HandButton(scenario._client,
                                get_hands,
                                btn['name'],
                                coord.fromWorldToPixelBbox(btn['bbox']),
                                angle_deg,
                                btn['timer'],
                                scenario.achieveCompletionEvent,
                                btn['completion_event'])
        # Dummy button
        else:
            button = HandButton(scenario._client,
                                get_hands,
                                btn['name'],
                                coord.fromWorldToPixelBbox(btn['bbox']),
                                angle_deg,
                                btn['timer'],
                                dummy)

        # Apply button characteristics
        if 'press_up_image' in btn:
            image = btn['press_up_image']
            if type(image) is dict:
                button.setPressUpImage(image['image'])
            else:
                button.setPressUpImage(image)

        if 'press_down_image' in btn:
            image = btn['press_down_image']
            if type(image) is dict:
                button.setPressDownImage(image['image'])
            else:
                button.setPressDownImage(image)

        if 'press_down_sound' in btn:
            button.setPressDownSound(btn['press_down_sound'])

        if 'press_end_sound' in btn:
            button.setPressDownEndSound(btn['press_end_sound'])

        if 'desired_work_hand' in btn:
            if btn['desired_work_hand'] == 'Recent':
                work_hand = scenario.recent_hand
            else:
                work_hand = btn['desired_work_hand']
            button.setDesiredWorkHand(work_hand)

        if 'handle_grasp' in btn:
            button.setHandleGrasp(btn['handle_grasp'])

        # Handle display delay
        if 'display_delay' in btn:
            button.display_delay = btn['display_delay']
        else:
            button.display_delay = 0

        # Append to list
        scenario.buttons.append(button)

    # Add buttons
    if 'buttons' in step_info:
        for btn in step_info['buttons']:
            # Handle json import
            if isinstance(btn, str) and btn.startswith("import"):
                import_path = btn.split(":")[1]
                # Load buttons file
                with open(import_path, encoding='utf8') as f:
                    import_buttons = json.load(f)
                    for import_button in import_buttons:
                        addButton(import_button)
                continue

            # Add a button
            addButton(btn)

    # =========== Create detection object(s) ============
    scenario.detections = []

    if 'detections' in step_info:
        for det in step_info['detections']:
            detection = ObjectButton(scenario._client,
                                        coord.fromWorldToPixelBbox(det['bbox']),
                                        det['class'],
                                        det['timer'],
                                        scenario.achieveCompletionEvent,
                                        det['completion_event'])

            # Apply detection characteristics
            if 'default_image' in det:
                default_image = det['default_image']
                detection.setDefaultImage(default_image['path'],
                                            coord.fromWorldToPixelBbox(default_image['bbox']))
            if 'incorrect_image' in det:
                incorrect_image = det['incorrect_image']
                detection.setIncorrectImage(incorrect_image['path'],
                                            coord.fromWorldToPixelBbox(incorrect_image['bbox']))
            if 'correct_image' in det:
                correct_image = det['correct_image']
                detection.setCorrectImage(correct_image['path'],
                                            coord.fromWorldToPixelBbox(correct_image['bbox']))

            # Append to list
            scenario.detections.append(detection)

    # =========== Create image object(s) ============
    scenario.images = {}

    if "images" in step_info:
        for image_conf in step_info["images"]:
            if not isinstance(image_conf, dict):
                logger.error("Invalid image definition. Expected dictionary.")
                continue

            try:
                image_path = image_conf["path"]
                image_bbox_world = image_conf["bbox"]
            except KeyError as missing_key:
                logger.error(f"Image definition missing required key: {missing_key}.")
                continue

            name = image_conf.get("name")
            if not name:
                name = f"image_{len(scenario.images)}"
            elif name in scenario.images:
                logger.warning(
                    f"Image name '{name}' already exists in this step. Skipping duplicated definition."
                )
                continue

            try:
                image_bbox_pixel = coord.fromWorldToPixelBbox(image_bbox_world)
                angle_deg = image_conf.get("angle", 0)
                display_delay = image_conf.get("display_delay", 0)
                visible = image_conf.get("visible", True)

                scenario_image = Image(
                    scenario._client,
                    name,
                    image_path,
                    image_bbox_pixel,
                    angle_deg=angle_deg,
                    display_delay=display_delay,
                    visible=visible,
                )
            except (ValueError, FileNotFoundError) as err:
                logger.error(str(err))
                continue

            scenario.images[name] = scenario_image

    # =========== Create LPIPS object(s) ============
    scenario.lpips_objects = {}
    if 'lpips' in step_info:
        for lpips_conf in step_info['lpips']:
            if not isinstance(lpips_conf, dict):
                logger.error("Invalid lpips definition. Expected dictionary.")
                continue
            
            try:
                name = lpips_conf['name']
                bbox = lpips_conf['bbox']
                reference = lpips_conf['reference']
                threshold = lpips_conf['threshold']
                completion_event = lpips_conf['completion_event']
                autocheck_interval = lpips_conf.get('autocheck_interval')
                autocheck_hand_free_interval = lpips_conf.get('autocheck_hand_free_interval')
                use_mask = lpips_conf.get('use_mask', True)
                mask_margin = lpips_conf.get('mask_margin')
                use_alignment = lpips_conf.get('use_alignment', True)
                alignment_margin = lpips_conf.get('alignment_margin')
                alignment_method = lpips_conf.get('alignment_method', 'multi_offset')
                alignment_search_step = lpips_conf.get('alignment_search_step', 4)
                alignment_blur_sigma = lpips_conf.get('alignment_blur_sigma', 0.5)
            except KeyError as missing_key:
                logger.error(f"LPIPS definition missing required key: {missing_key}.")
                continue
            
            lpips_obj = LPIPSObject(
                scenario._client,
                name,
                bbox,
                reference,
                threshold,
                scenario.achieveCompletionEvent,
                completion_event,
                autocheck_interval=autocheck_interval,
                autocheck_hand_free_interval=autocheck_hand_free_interval,
                use_mask=use_mask,
                mask_margin=mask_margin,
                use_alignment=use_alignment,
                alignment_margin=alignment_margin,
                alignment_method=alignment_method,
                alignment_search_step=alignment_search_step,
                alignment_blur_sigma=alignment_blur_sigma)
            
            scenario.lpips_objects[name] = lpips_obj
            scenario.completion_events[f"lpips:{name}"] = False

    # =========== Create popup object(s) ============
    scenario.popups = {}
    if 'popups' in step_info:
        for pop in step_info['popups']:
            if 'completion_event' in pop:
                # Create a popup object with completion event(Set achieveCompletionEvent as callback)
                popup = Popup(scenario._client,
                                pop['name'], pop['image'], coord.fromWorldToPixelBbox(pop['bbox']),
                                pop['timer'], scenario.achieveCompletionEvent, pop['completion_event'])
            else:
                # Create a popup object with dummy callback
                popup = Popup(scenario._client,
                                pop['name'], pop['image'], coord.fromWorldToPixelBbox(pop['bbox']),
                                pop['timer'], dummy)

            if 'sound' in pop:
                popup.setPopupSound(pop['sound'])

            # Append to list
            scenario.popups[pop['name']] = popup
            scenario.completion_events[f"popup:{pop['name']}"] = False

    # =========== Create text object(s) ============
    scenario.texts = {}
    if "texts" in step_info:
        for text_conf in step_info["texts"]:
            if not isinstance(text_conf, dict):
                logger.error("Invalid text definition. Expected dictionary.")
                continue

            try:
                text_name = text_conf["name"]
                text_value = text_conf.get("text", "")
                text_position = text_conf["position"]
                text_size = text_conf["size"]
                text_align = text_conf.get("align")
                text_color = text_conf.get("color")
            except KeyError as missing_key:
                logger.error(f"Text definition missing required key: {missing_key}.")
                continue

            if text_color is not None:
                if isinstance(text_color, list):
                    text_color = tuple(text_color)
                if isinstance(text_color, tuple):
                    if len(text_color) != 3:
                        logger.error("Text color must have exactly three components.")
                        continue
                elif isinstance(text_color, str):
                    text_color = text_color.strip()
                else:
                    logger.error(
                        f"Unsupported color definition for text '{text_name}'."
                    )
                    continue

            try:
                scenario_text = Text(
                    scenario._client,
                    text_name,
                    text_value,
                    text_position,
                    text_size,
                    text_align,
                    text_color,
                )
            except ValueError as err:
                logger.error(str(err))
                continue

            scenario.texts[text_name] = scenario_text

    logger.info(f"Go to step {step_info['id']} - {step_info['name']}")

    # Load pygame resources
    scenario.loadBackground()
    scenario.loadGuideSound()
    scenario.loadGuideVideo()

    scenario.running_state = STATE.RUNNING

    return True
