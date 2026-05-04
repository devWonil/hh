# game_flow.py

from enum import Enum
from game_engine import GameEngine, GamePhase


class GameState(Enum):
    WAITING_DICE = "waiting_dice"
    MOVING       = "moving"
    BUY_PROMPT   = "buy_prompt"
    ENDED        = "ended"


class GameFlowManager:
    """
    GameEngine을 감싸 monopoly_integration이 필요로 하는
    인터페이스를 제공하는 플로우 매니저.
    """

    def __init__(self, num_players: int = 4):
        self.engine = GameEngine()
        self.num_players = num_players
        self.game_state = GameState.WAITING_DICE
        self.reset_game()

    # ── 초기화 ────────────────────────────────────────────────────

    def reset_game(self):
        self.engine.setup(self.num_players)
        self.game_state = GameState.WAITING_DICE

    # ── 플레이어 조회 ─────────────────────────────────────────────

    def get_current_player(self):
        """현재 턴 플레이어 객체 반환"""
        return self.engine.current_player

    def get_all_players(self):
        return self.engine.players

    # ── 이동 ──────────────────────────────────────────────────────

    def move_player(self, steps: int):
        """
        주사위 값을 받아 현재 플레이어를 이동시킨다.
        내부적으로 engine.roll_dice() 대신 직접 이동 처리.
        (AR에서는 실물 주사위를 감지해 steps를 넘겨주므로)
        """
        player = self.engine.current_player
        self.engine._move_player(player, steps)
        self.engine.phase = GamePhase.ACTION
        self.engine._apply_square_effect(player)
        self.game_state = GameState.MOVING

    # ── 구매 ──────────────────────────────────────────────────────

    def purchase_property(self, decision: bool) -> bool:
        """
        구매 여부를 engine에 전달한다.
        Returns True if purchase succeeded.
        """
        if self.engine.phase != GamePhase.BUY_PROMPT:
            return False
        self.engine.decide_buy(decision)
        self.game_state = GameState.WAITING_DICE
        return decision

    @property
    def property_prices(self) -> dict:
        """square_index → price 매핑 반환 (monopoly_integration 호환용)"""
        return {
            i: sq["price"]
            for i, sq in enumerate(self.engine.board)
            if sq.get("price") is not None
        }

    # ── 파산 / 종료 체크 ──────────────────────────────────────────

    def check_bankruptcy(self):
        """
        파산자 정리 후 생존자가 1명이면 해당 Player 객체 반환.
        게임 계속이면 None 반환.
        """
        active = self.engine._active_players()
        if len(active) == 1:
            self.game_state = GameState.ENDED
            return active[0]
        return None

    # ── 턴 종료 ───────────────────────────────────────────────────

    def end_turn(self):
        """다음 플레이어로 턴을 넘긴다."""
        if self.engine.phase not in (GamePhase.BUY_PROMPT, GamePhase.ENDED):
            self.engine._advance_turn()
        self.game_state = GameState.WAITING_DICE

    # ── 상태 스냅샷 ───────────────────────────────────────────────

    def get_game_status(self) -> dict:
        state = self.engine.get_state()
        state["game_state"] = self.game_state.value
        return state

