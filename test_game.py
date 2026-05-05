# test_game.py
# game_engine.py + game_flow.py만 있으면 실행 가능한 독립 테스트
# YOLO/카메라 없이 콘솔에서 직접 입력으로 게임 진행

import sys
sys.path.insert(0, '.')   # game_engine.py, game_flow.py와 같은 폴더에 위치시킬 것

from game_flow import GameFlowManager, GameState
from game_engine import GamePhase


def print_status(gf: GameFlowManager):
    state = gf.get_game_status()
    print()
    print("─" * 45)
    for p in state["players"]:
        tag = "▶" if p["id"] - 1 == state["turn"] else " "
        bankrupt = " [파산]" if p["is_bankrupt"] else ""
        props = len(p["owned_props"])
        print(f" {tag} {p['name']:8s} | {p['money']:>10,}원 | 부동산 {props}칸{bankrupt}")
    print("─" * 45)


def print_board_position(gf: GameFlowManager):
    state = gf.get_game_status()
    player = gf.get_current_player()
    sq = gf.engine.board[player.position]
    owner_id = gf.engine._owner_of(player.position)
    owner_str = f" (소유: Player{owner_id})" if owner_id else " (미분양)"
    print(f"  현재 위치: [{player.position}] {sq['name']}{owner_str}")


def get_int_input(prompt: str, lo: int, hi: int) -> int:
    while True:
        try:
            v = int(input(prompt))
            if lo <= v <= hi:
                return v
            print(f"  {lo}~{hi} 사이 숫자를 입력하세요.")
        except ValueError:
            print("  숫자를 입력하세요.")


def main():
    print("=" * 45)
    print("   AR 부루마블 — 콘솔 테스트 모드")
    print("=" * 45)

    num = get_int_input("플레이어 수 (2~4): ", 2, 4)
    gf = GameFlowManager(num_players=num)

    print(f"\n{num}명으로 게임 시작! (시작 자금 2,000,000원)\n")

    turn_count = 0

    while True:
        state = gf.get_game_status()

        # 게임 종료 체크
        if state["phase"] == GamePhase.ENDED.value or state["winner"]:
            print("\n" + "=" * 45)
            print(f"  게임 종료! 🏆 우승자: {state['winner']}")
            print("=" * 45)
            print_status(gf)
            break

        player = gf.get_current_player()

        # 파산 플레이어 자동 스킵
        if player.is_bankrupt:
            gf.end_turn()
            continue

        turn_count += 1
        print_status(gf)
        print(f"\n  [{turn_count}턴] {player.name}의 차례")

        # ── 주사위 입력 ──────────────────────────────
        print("  주사위 입력 방법:")
        print("  [1] 직접 입력  [2] 자동 랜덤")
        mode = get_int_input("  선택: ", 1, 2)

        if mode == 1:
            d1 = get_int_input("  주사위1 (1~6): ", 1, 6)
            d2 = get_int_input("  주사위2 (1~6): ", 1, 6)
            steps = d1 + d2
            print(f"  주사위: {d1} + {d2} = {steps}")
        else:
            import random
            d1, d2 = random.randint(1, 6), random.randint(1, 6)
            steps = d1 + d2
            print(f"  주사위(자동): {d1} + {d2} = {steps}")

        # ── skip_turns 처리 ──────────────────────────
        if player.skip_turns > 0:
            player.skip_turns -= 1
            print(f"  무인도 결석! (남은 턴: {player.skip_turns})")
            gf.end_turn()
            continue

        # ── 이동 ────────────────────────────────────
        gf.move_player(steps)
        print_board_position(gf)

        # 이벤트 로그 출력
        for msg in gf.engine.event_log[-3:]:
            print(f"  → {msg}")

        # ── 구매 프롬프트 ────────────────────────────
        if gf.engine.phase == GamePhase.BUY_PROMPT:
            sq = gf.engine.board[player.position]
            print(f"\n  {sq['name']} 구매? (가격: {sq['price']:,}원 / 잔액: {player.money:,}원)")
            ans = input("  [y] 구매  [n] 포기: ").strip().lower()
            gf.purchase_property(ans == 'y')

        # ── 파산 체크 후 종료 여부 확인 ──────────────
        winner = gf.check_bankruptcy()
        if winner:
            print("\n" + "=" * 45)
            print(f"  게임 종료! 🏆 우승자: {winner.name}")
            print("=" * 45)
            print_status(gf)
            break

        gf.end_turn()


if __name__ == "__main__":
    main()