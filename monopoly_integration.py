# Integration module connecting SULIVAN with Monopoly Game Engine

import logging
import time
import threading
from typing import Dict, Optional

from game_flow import GameFlowManager, GameState
from dice_detection import DiceDetector
from building_detection import BuildingDetector

logger = logging.getLogger(__name__)


class MonopolyGameManager:
    """
    Main manager integrating YOLO detection, game flow, and UI updates
    """

    def __init__(self, sulivan_client):
        """
        Initialize Monopoly Game Manager

        Args:
            sulivan_client: SULIVAN_Client instance
        """
        self.client = sulivan_client

        # Initialize game components
        self.game_flow = GameFlowManager(num_players=4)
        self.dice_detector = DiceDetector(sulivan_client.yolo_thread)
        self.building_detector = BuildingDetector(sulivan_client.yolo_thread)

        # Game state tracking
        self.is_running = False
        self.current_frame = None
        self.dice_detected_event = threading.Event()
        self.building_verified_event = threading.Event()

        # Timing
        self.dice_detection_timeout = 5.0   # seconds
        self.building_detection_timeout = 3.0  # seconds

        logger.info("Monopoly Game Manager initialized")

    def start_game(self):
        """Start a new game"""
        self.is_running = True
        self.game_flow.reset_game()
        self.game_flow.game_state = GameState.WAITING_DICE
        logger.info("Game started!")

    def stop_game(self):
        """Stop the game"""
        self.is_running = False
        logger.info("Game stopped!")

    def get_game_status(self) -> Dict:
        """Get current game status"""
        return self.game_flow.get_game_status()

    # ==================== DICE DETECTION PHASE ====================

    def wait_for_dice_roll(self, timeout: float = None) -> Optional[int]:
        """
        Wait for dice to be rolled and detected

        Args:
            timeout: Timeout in seconds

        Returns:
            Total dice sum or None if timeout
        """
        if timeout is None:
            timeout = self.dice_detection_timeout

        start_time = time.time()
        self.dice_detector.reset()

        logger.info("Waiting for dice roll...")

        while time.time() - start_time < timeout:
            frame = self.client.camera_thread.getImage()
            if frame is not None:
                success, dice_values = self.dice_detector.detect_dice(frame)

                if success and sum(dice_values) > 0:
                    dice_sum = self.dice_detector.get_dice_sum()
                    logger.info(
                        f"Dice detected: {dice_values[0]} + {dice_values[1]} = {dice_sum}"
                    )
                    return dice_sum

            time.sleep(0.05)  # Small delay to reduce CPU usage

        logger.warning("Dice detection timeout")
        return None

    # ==================== MOVEMENT PHASE ====================

    def execute_movement(self, steps: int):
        """
        Execute player movement

        Args:
            steps: Number of steps to move
        """
        logger.info(f"Moving player by {steps} steps")
        self.game_flow.move_player(steps)

        player = self.game_flow.get_current_player()
        logger.info(f"Player at square {player.position}")

    # ==================== BUILDING DETECTION PHASE ====================

    def verify_building_placement(self, timeout: float = None) -> bool:
        """
        Verify if building is correctly placed at current square

        Args:
            timeout: Timeout in seconds

        Returns:
            True if building detected and placed correctly
        """
        if timeout is None:
            timeout = self.building_detection_timeout

        start_time = time.time()
        player = self.game_flow.get_current_player()
        square_id = player.position

        logger.info(f"Verifying building placement at square {square_id}")

        while time.time() - start_time < timeout:
            frame = self.client.camera_thread.getImage()
            if frame is not None:
                self.building_detector.detect_buildings(frame)

                has_building = self.building_detector.get_building_at_square(square_id)

                if has_building:
                    verified = self.building_detector.verify_building_placement(square_id)
                    if verified:
                        logger.info(f"Building verified at square {square_id}")
                        return True

            time.sleep(0.05)

        logger.info(f"No building detected at square {square_id}")
        return False

    # ==================== PURCHASE DECISION PHASE ====================

    def prompt_building_purchase(self) -> bool:
        """
        Prompt player to decide on property purchase

        Returns:
            True if player chooses to buy
        """
        player = self.game_flow.get_current_player()
        square_id = player.position
        price = self.game_flow.property_prices.get(square_id, 1_000_000)

        # UI 레이어에서 실제 다이얼로그로 교체 필요
        logger.info(
            f"Player {player.player_id} can buy property at square {square_id} "
            f"for {price:,}원"
        )

        # Placeholder: UI 연결 전까지 항상 구매
        return True

    def handle_property_purchase(self, purchase_decision: bool) -> bool:
        """
        Handle property purchase based on player decision

        Args:
            purchase_decision: True if player wants to buy

        Returns:
            True if purchase was successful
        """
        return self.game_flow.purchase_property(purchase_decision)

    # ==================== END TURN PHASE ====================

    def check_game_end(self) -> Optional[str]:
        """
        Check if game has ended

        Returns:
            Winner name if game ended, None otherwise
        """
        winner = self.game_flow.check_bankruptcy()
        if winner:
            logger.info(f"Game ended! Winner: Player {winner.player_id}")
            return f"Player {winner.player_id}"
        return None

    def end_turn(self):
        """End current turn and move to next player"""
        self.game_flow.end_turn()

    # ==================== MAIN GAME LOOP ====================

    def play_one_turn(self) -> bool:
        """
        Execute one complete game turn

        Returns:
            False if game has ended, True otherwise
        """
        if not self.is_running:
            return False

        player = self.game_flow.get_current_player()
        logger.info(f"=== Turn Start: Player {player.player_id} ===")

        # Phase 1: Wait for dice roll
        dice_sum = self.wait_for_dice_roll()
        if dice_sum is None:
            logger.error("Dice detection failed, skipping turn")
            self.end_turn()
            return True

        # Phase 2: Move player
        self.execute_movement(dice_sum)

        # Phase 3: Check for building placement (AR 실물 감지)
        has_building = self.verify_building_placement()

        # Phase 4: Handle property purchase if building exists
        if has_building:
            should_buy = self.prompt_building_purchase()
            self.handle_property_purchase(should_buy)

        # Phase 5: Check for bankruptcy / game end
        winner = self.check_game_end()
        if winner:
            return False  # Game ended

        # Phase 6: End turn
        self.end_turn()
        logger.info("=== Turn End ===\n")

        return True

    def play_game_loop(self):
        """Main game loop - play until game ends"""
        self.start_game()

        while self.is_running:
            game_continues = self.play_one_turn()

            if not game_continues:
                logger.info("Game finished!")
                self.stop_game()
                break

            time.sleep(1)  # Brief pause between turns

