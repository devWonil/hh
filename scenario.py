# Modified scenario.py for AR boardgame loop support


class Game:
    def __init__(self):
        self.steps: list         = []   # 수정: play()에서 참조하는데 정의가 없었음
        self.current_step: int   = 0
        self.loop_active: bool   = False
        self.loop_condition: bool = False

    def add_step(self, step) -> None:
        """스텝을 순서대로 추가한다."""
        self.steps.append(step)

    def reset(self) -> None:
        """게임을 처음 상태로 초기화한다."""
        self.current_step = 0
        self.loop_active = False
        self.loop_condition = False

    def play(self) -> None:
        """모든 스텝을 순서대로 실행한다."""
        while self.current_step < len(self.steps):
            if self.loop_active:
                if not self.check_loop_condition():
                    # 조건 불충족 → 루프 탈출 후 다음 스텝으로 이동
                    self.loop_active = False
                    self.current_step += 1
                else:
                    self.handle_loop()
                    # handle_loop() 내부에서 stop_loop()이 호출된 경우 다음 스텝으로 이동
                    if not self.loop_active:
                        self.current_step += 1
            else:
                self.execute_step(self.current_step)
                self.current_step += 1

    def execute_step(self, step_index: int) -> None:
        """step_index에 해당하는 스텝을 실행한다."""
        if step_index < 0 or step_index >= len(self.steps):
            raise IndexError(f"유효하지 않은 스텝 인덱스: {step_index}")
        step = self.steps[step_index]
        # step 객체가 callable이면 호출, 아니면 그냥 반환
        if callable(step):
            step()

    def handle_loop(self) -> None:
        """루프 중 현재 스텝을 재실행한다."""
        self.execute_step(self.current_step)

    def check_loop_condition(self) -> bool:
        """루프 지속 조건을 반환한다. 외부에서 loop_condition을 설정해 제어한다."""
        return self.loop_condition

    def start_loop(self) -> None:
        """현재 스텝에서 루프를 시작한다."""
        self.loop_active = True

    def stop_loop(self) -> None:
        """루프를 즉시 종료하고 다음 스텝으로 넘어간다."""
        self.loop_active = False
        self.loop_condition = False
