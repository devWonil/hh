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
    FESTIVAL   = "festival"    # 축제: 전원에게 돈 받기
    SPACE      = "space"       # 우주여행: 전원에게 돈 지불


class GamePhase(Enum):
    WAITING     = "waiting"
    ROLL        = "roll"
    ACTION      = "action"
    BUY_PROMPT  = "buy_prompt"
    SELL_PROMPT = "sell_prompt"   # 통행료 부족 시 매각 여부 결정
    ENDED       = "ended"


BOARD_SIZE     = 28
STARTING_MONEY = 2_000_000
SALARY         = 200_000
JAIL_SQUARE    = 9
TAX_AMOUNT     = 1_500_000
TRANSPORT_FEE  = 1_500_000
FESTIVAL_EARN  = 300_000     # 축제: 다른 플레이어 1인당 받는 금액
SPACE_PAY      = 300_000     # 우주여행: 다른 플레이어 1인당 지불 금액


BOARD_SQUARES = [
    # idx  이름              타입                     가격         임대료       그룹
    {"name": "시작",        "type": SquareType.START},                                               # 0
    {"name": "베이징",      "type": SquareType.PROPERTY,  "price": 500_000,  "rent": 500_000,  "group": 0},  # 1
    {"name": "싱가포르",    "type": SquareType.PROPERTY,  "price": 600_000,  "rent": 600_000,  "group": 0},  # 2
    {"name": "기부주기",    "type": SquareType.TAX},                                                # 3
    {"name": "카이로",      "type": SquareType.PROPERTY,  "price": 700_000,  "rent": 700_000,  "group": 1},  # 4
    {"name": "이스탄불",    "type": SquareType.PROPERTY,  "price": 800_000,  "rent": 800_000,  "group": 1},  # 5
    {"name": "독도",        "type": SquareType.TRANSPORT, "price": 500_000,  "rent": 500_000},               # 6
    {"name": "아테네",      "type": SquareType.PROPERTY,  "price": 900_000,  "rent": 900_000,  "group": 1},  # 7
    {"name": "스톡홀름",    "type": SquareType.PROPERTY,  "price": 1_000_000,"rent": 1_000_000,"group": 2},  # 8
    {"name": "무인도",      "type": SquareType.JAIL_VISIT},                                         # 9
    {"name": "베른",        "type": SquareType.PROPERTY,  "price": 1_100_000,"rent": 1_100_000,"group": 2},  # 10
    {"name": "하와이",      "type": SquareType.TRANSPORT, "price": 600_000,  "rent": 600_000},               # 11
    {"name": "베를린",      "type": SquareType.PROPERTY,  "price": 1_200_000,"rent": 1_200_000,"group": 2},  # 12
    {"name": "오타와",      "type": SquareType.PROPERTY,  "price": 1_300_000,"rent": 1_300_000,"group": 3},  # 13
    {"name": "축제",        "type": SquareType.FESTIVAL},                                           # 14
    {"name": "부에노스아이레스","type": SquareType.PROPERTY,"price": 1_400_000,"rent": 1_400_000,"group": 3}, # 15
    {"name": "상파울로",    "type": SquareType.PROPERTY,  "price": 1_500_000,"rent": 1_500_000,"group": 3},  # 16
    {"name": "시드니",      "type": SquareType.PROPERTY,  "price": 1_600_000,"rent": 1_600_000,"group": 4},  # 17
    {"name": "기부받기",    "type": SquareType.FREE},                                               # 18
    {"name": "도쿄",        "type": SquareType.PROPERTY,  "price": 1_700_000,"rent": 1_700_000,"group": 4},  # 19
    {"name": "부산",        "type": SquareType.TRANSPORT, "price": 700_000,  "rent": 700_000},               # 20
    {"name": "파리",        "type": SquareType.PROPERTY,  "price": 1_800_000,"rent": 1_800_000,"group": 4},  # 21
    {"name": "로마",        "type": SquareType.PROPERTY,  "price": 1_900_000,"rent": 1_900_000,"group": 4},  # 22
    {"name": "우주여행",    "type": SquareType.SPACE},                                              # 23
    {"name": "런던",        "type": SquareType.PROPERTY,  "price": 2_000_000,"rent": 2_000_000,"group": 5},  # 24
    {"name": "제주도",      "type": SquareType.TRANSPORT, "price": 800_000,  "rent": 800_000},               # 25
    {"name": "뉴욕",        "type": SquareType.PROPERTY,  "price": 2_100_000,"rent": 2_100_000,"group": 5},  # 26
    {"name": "서울",        "type": SquareType.PROPERTY,  "price": 2_200_000,"rent": 2_200_000,"group": 5},  # 27
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

        # SELL_PROMPT 단계에서 사용하는 임시 상태
        self._pending_fee:      int    = 0
        self._pending_receiver: Player = None
        self._pending_label:    str    = ""
        self._pending_acquire:  bool   = False
        self._pending_space_others: list  = []   # 우주여행 분배 대상

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
                self._transfer_money(player, self._get_player(owner_id), fee, label,
                                     allow_acquire=True)

        elif stype == SquareType.CHANCE:
            self._draw_chance(player)

        elif stype == SquareType.TAX:
            if TAX_AMOUNT > player.money and player.owned_props:
                # 세금이 잔액보다 크고 팔 땅이 있으면 매각 프롬프트
                self._pending_fee      = TAX_AMOUNT
                self._pending_receiver = None   # 세금은 받는 플레이어 없음
                self._pending_label    = "세금"
                self._pending_acquire  = False
                sellable = self._get_sellable_props(player)
                names = ", ".join(
                    f"{self.board[i]['name']}({self.board[i]['price']//2:,}원)"
                    for i in sellable
                )
                self._log(f"{player.name} 세금 잔액 부족! 매각 가능 땅: {names}")
                self._log("매각하시겠습니까? (매각가: 원가의 50%)")
                self.phase = GamePhase.SELL_PROMPT
            else:
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
            self._log(f"{player.name} 기부받기 — 무료 휴식.")
            self._advance_turn()

        elif stype == SquareType.FESTIVAL:
            # 축제: 생존한 다른 플레이어 1인당 FESTIVAL_EARN 받기
            others = [p for p in self._active_players() if p.player_id != player.player_id]
            total = 0
            for other in others:
                earn = min(FESTIVAL_EARN, other.money)
                other.money  -= earn
                player.money += earn
                total += earn
                self._log(f"{other.name} → {player.name} 축제 분담금 -{earn:,}원")
            self._log(f"{player.name} 축제! 총 +{total:,}원 획득 (잔액 {player.money:,}원)")
            # 분담금 낸 플레이어 파산 체크
            for other in others:
                self._check_bankrupt(other)
            self._advance_turn()

        elif stype == SquareType.SPACE:
            # 우주여행: 생존한 다른 플레이어 1인당 SPACE_PAY 지불
            others = [p for p in self._active_players() if p.player_id != player.player_id]
            total_needed = SPACE_PAY * len(others)
            if total_needed > player.money and player.owned_props:
                # 잔액 부족 시 매각 프롬프트
                self._pending_fee      = total_needed
                self._pending_receiver = None   # 우주여행은 다수에게 분배 → 특수 처리
                self._pending_label    = "우주여행 비용"
                self._pending_acquire  = False
                self._pending_space_others = others   # 분배 대상 저장
                sellable = self._get_sellable_props(player)
                names = ", ".join(
                    f"{self.board[i]['name']}({self.board[i]['price']//2:,}원)"
                    for i in sellable
                )
                self._log(f"{player.name} 우주여행 비용 부족! 매각 가능 땅: {names}")
                self._log("매각하시겠습니까? (매각가: 원가의 50%)")
                self.phase = GamePhase.SELL_PROMPT
            else:
                self._pay_space(player, others)

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
                        amount: int, label: str,
                        allow_acquire: bool = False) -> None:
        # 통행료가 잔액보다 크고 팔 땅이 있으면 매각 프롬프트 먼저
        if amount > payer.money and payer.owned_props:
            self._pending_fee      = amount
            self._pending_receiver = receiver
            self._pending_label    = label
            self._pending_acquire  = allow_acquire
            sellable = self._get_sellable_props(payer)
            names = ", ".join(f"{self.board[i]['name']}({self.board[i]['price']//2:,}원)" for i in sellable)
            self._log(f"{payer.name} 잔액 부족! 매각 가능 땅: {names}")
            self._log("매각하시겠습니까? (매각가: 원가의 50%)")
            self.phase = GamePhase.SELL_PROMPT
            return

        actual = min(amount, payer.money)
        payer.money    -= actual
        receiver.money += actual
        self._log(f"{payer.name} {label} -{actual:,}원 → {receiver.name} "
                  f"(잔액 {payer.money:,}원)")
        self._check_bankrupt(payer)

        # 통행료 지불 후 해당 땅 인수 가능 여부 확인
        if allow_acquire and not payer.is_bankrupt:
            sq = self.board[payer.position]
            if payer.money >= sq["price"]:
                self._log(f"{sq['name']} 인수 가능 (가격 {sq['price']:,}원). 구매하시겠습니까?")
                self.phase = GamePhase.BUY_PROMPT
                return

        self._advance_turn()


    def _pay_space(self, player, others: list) -> None:
        """우주여행: player가 others 각각에게 SPACE_PAY 지불."""
        total = 0
        for other in others:
            pay = min(SPACE_PAY, player.money)
            player.money -= pay
            other.money  += pay
            total += pay
            self._log(f"{player.name} → {other.name} 우주여행 비용 -{pay:,}원")
        self._log(f"{player.name} 우주여행! 총 -{total:,}원 (잔액 {player.money:,}원)")
        self._check_bankrupt(player)
        self._advance_turn()

    def _get_sellable_props(self, player: Player) -> list[int]:
        """매각 가능한 땅 목록 반환 (소유 땅 전체)."""
        return list(player.owned_props)

    def decide_sell(self, sell: bool, prop_idx: int = None) -> dict:
        """
        SELL_PROMPT 단계에서 매각 여부를 결정합니다.

        Args:
            sell     : True면 매각, False면 매각 거부(그냥 파산 처리)
            prop_idx : 매각할 땅의 보드 인덱스 (sell=True일 때 필수)

        Returns:
            get_state() 스냅샷
        """
        if self.phase != GamePhase.SELL_PROMPT:
            raise RuntimeError("지금은 매각 결정 단계가 아닙니다.")

        player   = self.current_player
        receiver = self._pending_receiver
        amount   = self._pending_fee
        label    = self._pending_label
        acquire  = self._pending_acquire

        def _pay(p, amt, recv, lbl, reason=""):
            """실제 금액 지불 처리. recv가 None이면 세금(소각)."""
            actual = min(amt, p.money)
            p.money -= actual
            if recv is not None:
                recv.money += actual
                self._log(f"{p.name} {lbl} -{actual:,}원 → {recv.name} (잔액 {p.money:,}원){reason}")
            else:
                self._log(f"{p.name} {lbl} -{actual:,}원 (잔액 {p.money:,}원){reason}")
            self._check_bankrupt(p)

        if sell and prop_idx is not None:
            if prop_idx not in player.owned_props:
                raise ValueError(f"[{prop_idx}]는 {player.name}의 소유 땅이 아닙니다.")

            sell_price = self.board[prop_idx]["price"] // 2
            player.owned_props.remove(prop_idx)
            player.money += sell_price
            self._log(f"{player.name} [{self.board[prop_idx]['name']}] 매각 +{sell_price:,}원 (잔액 {player.money:,}원)")

            if player.money >= amount:
                # 매각 후 잔액 충분 → 지불
                if self._pending_space_others:
                    # 우주여행: 다수에게 분배
                    self._pay_space(player, self._pending_space_others)
                else:
                    _pay(player, amount, receiver, label)
                    # 인수 가능 여부 확인 (통행료인 경우만)
                    if acquire and not player.is_bankrupt:
                        sq = self.board[player.position]
                        if player.money >= sq["price"]:
                            self._log(f"{sq['name']} 인수 가능 (가격 {sq['price']:,}원). 구매하시겠습니까?")
                            self.phase = GamePhase.BUY_PROMPT
                            return self.get_state()
                    self._advance_turn()
            else:
                # 매각해도 여전히 부족
                if player.owned_props:
                    sellable = self._get_sellable_props(player)
                    names = ", ".join(
                        f"{self.board[i]['name']}({self.board[i]['price']//2:,}원)"
                        for i in sellable
                    )
                    self._log(f"아직 부족합니다. 추가 매각 가능 땅: {names}")
                    self.phase = GamePhase.SELL_PROMPT
                else:
                    # 더 팔 땅 없음 → 있는 돈만큼 내고 파산
                    if self._pending_space_others:
                        self._pay_space(player, self._pending_space_others)
                    else:
                        _pay(player, amount, receiver, label, " (잔액 부족, 파산)")
                    self._advance_turn()
        else:
            # 매각 거부 → 있는 돈만큼만 내고 파산
            if self._pending_space_others:
                self._pay_space(player, self._pending_space_others)
            else:
                _pay(player, amount, receiver, label, " (매각 거부, 파산)")
            self._advance_turn()

        # 임시 상태 정리
        self._pending_fee          = 0
        self._pending_receiver     = None
        self._pending_label        = ""
        self._pending_acquire      = False
        self._pending_space_others = []

        return self.get_state()

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
