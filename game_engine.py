import random
from enum import Enum


class SquareType(Enum):
    START      = "start"
    PROPERTY   = "property"
    CHANCE     = "chance"
    TAX        = "tax"
    TRANSPORT  = "transport"
    JAIL_VISIT = "jail_visit"
    GO_TO_JAIL = "go_to_jail"
    FREE       = "free"


class GamePhase(Enum):
    WAITING    = "waiting"
    ROLL       = "roll"
    ACTION     = "action"
    BUY_PROMPT = "buy_prompt"
    ENDED      = "ended"


BOARD_SIZE     = 28
STARTING_MONEY = 2_000_000
SALARY         = 200_000
JAIL_SQUARE    = 6
TAX_AMOUNT     = 200_000
TRANSPORT_FEE  = 150_000


BOARD_SQUARES = [
    {"name": "출발",   "type": SquareType.START},
    {"name": "서울",   "type": SquareType.PROPERTY,  "price": 500_000,  "rent": 50_000,  "group": 0},
    {"name": "기회",   "type": SquareType.CHANCE},
    {"name": "부산",   "type": SquareType.PROPERTY,  "price": 600_000,  "rent": 60_000,  "group": 0},
    {"name": "세금",   "type": SquareType.TAX},
    {"name": "제주도", "type": SquareType.PROPERTY,  "price": 700_000,  "rent": 70_000,  "group": 1},
    {"name": "무인도", "type": SquareType.JAIL_VISIT},
    {"name": "대구",   "type": SquareType.PROPERTY,  "price": 800_000,  "rent": 80_000,  "group": 1},
    {"name": "공항1",  "type": SquareType.TRANSPORT, "price": 500_000,  "rent": 100_000},
    {"name": "광주",   "type": SquareType.PROPERTY,  "price": 900_000,  "rent": 90_000,  "group": 1},
    {"name": "주차장", "type": SquareType.FREE},
    {"name": "인천",   "type": SquareType.PROPERTY,  "price": 1_000_000,"rent": 100_000, "group": 2},
    {"name": "기회",   "type": SquareType.CHANCE},
    {"name": "대전",   "type": SquareType.PROPERTY,  "price": 1_100_000,"rent": 110_000, "group": 2},
    {"name": "공항2",  "type": SquareType.TRANSPORT, "price": 500_000,  "rent": 100_000},
    {"name": "울산",   "type": SquareType.PROPERTY,  "price": 1_200_000,"rent": 120_000, "group": 2},
    {"name": "경찰서", "type": SquareType.GO_TO_JAIL},
    {"name": "수원",   "type": SquareType.PROPERTY,  "price": 1_300_000,"rent": 130_000, "group": 3},
    {"name": "공항3",  "type": SquareType.TRANSPORT, "price": 500_000,  "rent": 100_000},
    {"name": "성남",   "type": SquareType.PROPERTY,  "price": 1_400_000,"rent": 140_000, "group": 3},
    {"name": "세금",   "type": SquareType.TAX},
    {"name": "안산",   "type": SquareType.PROPERTY,  "price": 1_500_000,"rent": 150_000, "group": 3},
    {"name": "기회",   "type": SquareType.CHANCE},
    {"name": "고양",   "type": SquareType.PROPERTY,  "price": 1_600_000,"rent": 160_000, "group": 4},
    {"name": "공항4",  "type": SquareType.TRANSPORT, "price": 500_000,  "rent": 100_000},
    {"name": "용인",   "type": SquareType.PROPERTY,  "price": 1_700_000,"rent": 170_000, "group": 4},
    {"name": "기회",   "type": SquareType.CHANCE},
    {"name": "화성",   "type": SquareType.PROPERTY,  "price": 1_800_000,"rent": 180_000, "group": 4},
]

assert len(BOARD_SQUARES) == BOARD_SIZE, \
    f"보드 칸 수 오류: {len(BOARD_SQUARES)} (기대값: {BOARD_SIZE})"


CHANCE_CARDS = [
    {"text": "출발점으로 이동! 월급 200,000원 받기",  "move_to": 0,           "bonus": SALARY},
    {"text": "무인도로 유배! 1턴 쉬기",               "move_to": JAIL_SQUARE, "skip": True},
    {"text": "세금 환급! 1,000,000원 받기",           "money":  1_000_000},
    {"text": "집수리비 납부 -500,000원",              "money": -500_000},
    {"text": "복권 당첨! 500,000원 받기",             "money":  500_000},
    {"text": "가장 가까운 공항으로 이동",              "nearest_transport": True},
]


class Player:
    def __init__(self, player_id: int):
        self.player_id   = player_id
        self.name        = f"Player{player_id}"
        self.money       = STARTING_MONEY
        self.position    = 0
        self.skip_turns  = 0
        self.is_bankrupt = False
        self.owned_props: list[int] = []

    def net_worth(self, board: list[dict]) -> int:
        return self.money + sum(board[i]["price"] for i in self.owned_props)

    def declare_bankrupt(self):
        self.is_bankrupt = True
        self.money = 0
        self.owned_props.clear()

    def __repr__(self):
        return (f"<Player {self.name} | pos={self.position} | "
                f"money={self.money:,} | props={self.owned_props}>")


class GameEngine:
    def __init__(self):
        self.board:    list[dict]       = BOARD_SQUARES
        self.players:  list[Player]     = []
        self.turn_idx: int              = 0
        self.phase:    GamePhase        = GamePhase.WAITING
        self.last_dice: tuple[int, int] = (0, 0)
        self.event_log: list[str]       = []

    # ── 공개 API ───────────────────────────────────────────────────────────

    def setup(self, num_players: int) -> None:
        if not (2 <= num_players <= 4):
            raise ValueError("플레이어 수는 2~4명이어야 합니다.")
        self.players   = [Player(i + 1) for i in range(num_players)]
        self.turn_idx  = 0
        self.phase     = GamePhase.ROLL
        self.event_log = []
        self.last_dice = (0, 0)
        self._log(f"게임 시작! {num_players}명 참가 / 시작 자금 {STARTING_MONEY:,}원")
        self._log(f"{self.current_player.name}의 턴입니다.")

    def roll_dice(self) -> dict:
        if self.phase != GamePhase.ROLL:
            raise RuntimeError(f"지금은 주사위를 굴릴 수 없습니다. (phase={self.phase})")

        player = self.current_player

        if player.skip_turns > 0:
            player.skip_turns -= 1
            self._log(f"{player.name} 무인도 결석 (남은 턴: {player.skip_turns})")
            self._advance_turn()
            return self.get_state()

        d1, d2 = random.randint(1, 6), random.randint(1, 6)
        self.last_dice = (d1, d2)
        self._log(f"{player.name} 주사위: {d1}+{d2}={d1+d2}")

        self._move_player(player, d1 + d2)
        self.phase = GamePhase.ACTION
        self._apply_square_effect(player)
        return self.get_state()

    def decide_buy(self, buy: bool) -> dict:
        if self.phase != GamePhase.BUY_PROMPT:
            raise RuntimeError("지금은 구매 결정 단계가 아닙니다.")

        player = self.current_player
        sq     = self.board[player.position]

        if buy:
            player.money -= sq["price"]
            player.owned_props.append(player.position)
            self._log(f"{player.name} → {sq['name']} 구매! (잔액 {player.money:,}원)")
        else:
            self._log(f"{player.name} 구매 포기.")

        self._advance_turn()
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "phase":     self.phase.value,
            "turn":      self.turn_idx,
            "last_dice": self.last_dice,
            "event_log": list(self.event_log),
            "winner":    self._find_winner(),
            "players": [
                {
                    "id":          p.player_id,
                    "name":        p.name,
                    "money":       p.money,
                    "position":    p.position,
                    "skip_turns":  p.skip_turns,
                    "is_bankrupt": p.is_bankrupt,
                    "owned_props": list(p.owned_props),
                    "net_worth":   p.net_worth(self.board),
                }
                for p in self.players
            ],
            "board": [
                {
                    "index": i,
                    "name":  sq["name"],
                    "type":  sq["type"].value,
                    "price": sq.get("price"),
                    "rent":  sq.get("rent"),
                    "owner": self._owner_of(i),
                }
                for i, sq in enumerate(self.board)
            ],
        }

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────

    @property
    def current_player(self) -> Player:
        return self.players[self.turn_idx]

    def _log(self, msg: str) -> None:
        self.event_log.append(msg)
        if len(self.event_log) > 20:
            self.event_log.pop(0)

    def _move_player(self, player: Player, steps: int) -> None:
        old_pos = player.position
        new_pos = (old_pos + steps) % BOARD_SIZE
        if (old_pos + steps) >= BOARD_SIZE:
            player.money += SALARY
            self._log(f"{player.name} 출발점 통과! +{SALARY:,}원 (잔액 {player.money:,}원)")
        player.position = new_pos
        self._log(f"{player.name} → [{new_pos}] {self.board[new_pos]['name']}")

    def _move_to(self, player: Player, target: int, give_salary: bool = False) -> None:
        if give_salary and target <= player.position:
            player.money += SALARY
            self._log(f"{player.name} 출발점 통과! +{SALARY:,}원")
        player.position = target
        self._log(f"{player.name} → [{target}] {self.board[target]['name']} (직접 이동)")

    def _apply_square_effect(self, player: Player) -> None:
        sq    = self.board[player.position]
        stype = sq["type"]

        if stype == SquareType.START:
            self._advance_turn()

        elif stype in (SquareType.PROPERTY, SquareType.TRANSPORT):
            owner_id = self._owner_of(player.position)
            if owner_id is None:
                self._log(f"{sq['name']} 미분양 (가격 {sq['price']:,}원). 구매하시겠습니까?")
                if player.money >= sq["price"]:
                    self.phase = GamePhase.BUY_PROMPT
                else:
                    self._log(f"{player.name} 잔액 부족으로 구매 불가.")
                    self._advance_turn()
            elif owner_id == player.player_id:
                self._log("내 땅입니다.")
                self._advance_turn()
            else:
                fee   = sq["rent"] if stype == SquareType.PROPERTY else TRANSPORT_FEE
                label = "임대료" if stype == SquareType.PROPERTY else "공항 이용료"
                self._transfer_money(player, self._get_player(owner_id), fee, label)

        elif stype == SquareType.CHANCE:
            self._draw_chance(player)

        elif stype == SquareType.TAX:
            player.money = max(0, player.money - TAX_AMOUNT)
            self._log(f"{player.name} 세금 -{TAX_AMOUNT:,}원 (잔액 {player.money:,}원)")
            self._check_bankrupt(player)
            self._advance_turn()

        elif stype == SquareType.JAIL_VISIT:
            self._log(f"{player.name} 무인도 방문 (단순 통과).")
            self._advance_turn()

        elif stype == SquareType.GO_TO_JAIL:
            self._log(f"{player.name} 경찰서! 무인도로 이동, 1턴 쉬기.")
            self._move_to(player, JAIL_SQUARE)
            player.skip_turns = 1
            self._advance_turn()

        elif stype == SquareType.FREE:
            self._log(f"{player.name} 주차장 — 무료 휴식.")
            self._advance_turn()

    def _draw_chance(self, player: Player) -> None:
        card = random.choice(CHANCE_CARDS)
        self._log(f"[기회 카드] {card['text']}")

        if "move_to" in card:
            self._move_to(player, card["move_to"], give_salary=bool(card.get("bonus")))
            if card.get("bonus"):
                player.money += card["bonus"]
            if card.get("skip"):
                player.skip_turns = 1

        if "money" in card:
            player.money = max(0, player.money + card["money"])
            self._log(f"{player.name} 잔액 → {player.money:,}원")

        if card.get("nearest_transport"):
            ports = [i for i, s in enumerate(self.board)
                     if s["type"] == SquareType.TRANSPORT]
            nxt = next((t for t in ports if t > player.position), ports[0])
            self._move_to(player, nxt, give_salary=nxt < player.position)

        self._check_bankrupt(player)
        self._advance_turn()

    def _transfer_money(self, payer: Player, receiver: Player,
                        amount: int, label: str) -> None:
        actual = min(amount, payer.money)
        payer.money    -= actual
        receiver.money += actual
        self._log(f"{payer.name} {label} -{actual:,}원 → {receiver.name} "
                  f"(잔액 {payer.money:,}원)")
        self._check_bankrupt(payer)
        self._advance_turn()

    def _check_bankrupt(self, player: Player) -> None:
        if player.money <= 0 and not player.is_bankrupt:
            player.declare_bankrupt()
            self._log(f"{player.name} 파산!")
            if len(self._active_players()) == 1:
                self._log(f"{self._active_players()[0].name} 우승!")
                self.phase = GamePhase.ENDED

    def _advance_turn(self) -> None:
        if self.phase == GamePhase.ENDED:
            return
        if len(self._active_players()) <= 1:
            self.phase = GamePhase.ENDED
            return
        for _ in range(len(self.players)):
            self.turn_idx = (self.turn_idx + 1) % len(self.players)
            if not self.players[self.turn_idx].is_bankrupt:
                break
        self.phase = GamePhase.ROLL
        self._log(f"── {self.current_player.name}의 턴 ──")

    def _active_players(self) -> list[Player]:
        return [p for p in self.players if not p.is_bankrupt]

    def _owner_of(self, idx: int) -> int | None:
        for p in self.players:
            if idx in p.owned_props:
                return p.player_id
        return None

    def _get_player(self, player_id: int) -> Player:
        return next(p for p in self.players if p.player_id == player_id)

    def _find_winner(self) -> str | None:
        if self.phase != GamePhase.ENDED:
            return None
        active = self._active_players()
        if len(active) == 1:
            return active[0].name
        return max(self.players, key=lambda p: p.net_worth(self.board)).name
