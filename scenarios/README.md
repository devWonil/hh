SULIVAN scenario 문법
=============

루트 디렉토리의 scenarios 폴더 아래에 **하위 폴더**로 시나리오를 저장하며, 구조는 아래와 같다.
* **(시나리오 이름)**
  * **assets** : 리소스 파일 저장
    * **backgrounds** : 배경 이미지
    * **images** : 부품 사진 등 배경 위에 표시되는 이미지
    * **sounds** : 가이드 음성
    * **videos** : 가이드 영상
  * **scenario.json** : 시나리오 구성 파일
  * **config.json** : 시나리오 공통적으로 사용되는 설정값 정의

루트 디렉토리의 assets 폴더에 있는 리소스도 사용할 수 있으며, 시나리오 asset에 같은 파일이 존재하면 시나리오의 파일이 우선된다.

좌표계 설명
--------------
본 시스템에는 총 3개의 좌표계가 존재한다. 모든 좌표계의 원점은 왼쪽 상단이다.
* **Camera**
  * 카메라 FOV(Field of View, 카메라가 보는 영역)를 기준으로 하는 좌표계 (단위: pixel)
* **World**
  * 작업 공간(책상) 평면을 기준으로 하는 좌표계 (단위: cm)
  * Pixel <-> World 좌표계 간은 Scale 변환임
* **Pixel**
  * 빔 프로젝터에 투사되는 영상의 좌표계 (단위: pixel)
  * Camera -> Pixel 좌표계 간의 변환 행렬은 config에 정의된 M_cp임

버튼 인식 범위, 이미지 표시 위치 등을 지정할 때 Bounding Box(bbox)라는 위치 입력 방법을 사용한다. bbox는 World 좌표계 **(단위: cm)** 를 기준으로 아래와 같이 입력한다.

* [X 위치, Y 위치, 가로 크기, 세로 크기]

예를 들어 어떠한 이미지를 [10, 20, 15, 10] 이라는 bbox로 표시한다면 화면 왼쪽 위에서부터 오른쪽으로 10cm, 아래쪽으로 20cm만큼 떨어진 위치에서 가로 15cm, 세로 10cm로 이미지를 표시한다.
위치에 음수를 입력하는 것도 가능하나, 프로젝터 표시 범위가 아니므로 실제 표시는 되지 않는다.

config.json 구조
--------------
config.json에는 Information Area(사용자 이름과 사진, 경과 시간)에 대한 설정이 저장된다.

시나리오 폴더에 config.json이 존재하지 않을 경우 scenarios/default_config.json 파일이 로드된다.

**주의:** 이곳의 설정값은 Pixel 좌표계를 기준으로 한다. (단위: pixel, 가로 1920, 세로 1080)
```
{
    "info_font_size": 60,                      # Information Area 글꼴 크기
    "name_font_size": 60,                      # 사용자 이름 글꼴 크기
    "pose_user_name": [1470, 30],              # 사용자 이름 위치
    "rect_user_photo": [1701, 11, 173, 174],   # 사진 bbox(X,Y,W,H)
    "pose_scenario_tooltip": [1470, 170],      # "총 시간: " 텍스트 위치
    "pose_scenario_runtime": [1710, 170],      # 시나리오 경과시간 값 위치
    "pose_step_tooltip": [1470, 250],          # "현재: " 텍스트 위치
    "pose_step_runtime": [1710, 250]           # 스텝 경과시간 값 위치
}
```

하나의 scenario는 여러 개의 step으로 구성된다.

각 step은 하나의 dictionary로 구성되며, button과 detection을 list로 포함한다.

scenario.json 구조
--------------
```
[
    { Step },      # 0번 step
    { Step },      # 1번 step
    [
        { Step },  # 1-1번 step
        { Step },  # 1-2번 step
        { Step }   # 1-3번 step
    ],
    { Step },      # 2번 step
    [
        { Step },  # 2-1번 step
        { Step },  # 2-2번 step
        { Step }   # 2-3번 step
    ]
]
```

Step 구조
--------------
``` 
{
  "name": 스텝 이름
  "background": 배경화면(단일 or 리스트),
  "guide_sound": step 시작 시 재생할 안내 멘트 파일을 지정하거나, "tts:{읽을 문장}" 을 입력하여 입력한 문장을 음성으로 출력 (선택)
  "guide_sound_interval": guide_sound를 재생할 반복 간격(초, 선택)
  "guide_video": step 내에 재생할 비디오 파일 지정(선택)
  "guide_video_bbox": 비디오 파일의 위치와 크기(X,Y,W,H) 지정
  "display_runtime": true로 설정하면 config.json에 정의된 위치에 Information Area 표시(사용자 정보, 경과시간 등)
  "buttons": [ {button 1}, {button 2}, ... ] // 버튼 정의
  "images": [ {image 1}, {image 2}, ... ] // 화면에 고정으로 표시할 이미지 정의
  "detections": [ {detection 1}, {detection 2}, ... ] // YOLO 인식 정의
  "lpips": [ {lpips 1}, {lpips 2}, ... ] // LPIPS 이미지 유사도 비교 정의
  "popups": [ {popup 1}, {popup 2} ] // 팝업 정의
  "texts": [ {text 1}, {text 2} ] // 텍스트 정의
  "completion_events": [ 만족해야 할 조건 리스트(이름을 str로 입력, :는 명령어 구분자로 사용되므로 이름에는 사용하지 말 것), 모든 조건이 만족되어야 다음 Step으로 넘어감 ]
}
```
* completion_events 항목에 ```delay:2000``` 처럼 ms 단위로 기다리는 시간을 입력하면 해당 시간만큼 기다렸다가 달성되는 조건이 생성된다. completion_events를 정의하지 않으면 자동으로 다음 스텝으로 넘어가지 않으며, action 버튼을 통해서만 이동할 수 있다.
* TTS를 사용할 경우 EdgeTTS 모듈을 통해 텍스트를 작성하며, config.py에서 세부 설정을 할 수 있다. 한 번 읽은 문장은 캐시에 저장되어 다음 번부터는 빠르게 로드되며, config.py에서 설정을 변경하였다면 .cache 폴더를 삭제하여야 반영된다.
* 파일 경로를 지정할 때는 assets 폴더는 입력하지 않는다. (예: ```image/1.png```)
* buttons 항목에는 ```import:불러올 json 파일``` 과 같은 방법으로 다른 json 파일에 정의된 버튼을 불러올 수 있다.
* images 항목을 사용하면 지정한 위치(bbox)에 원하는 이미지를 회전(angle)한 상태로 표시할 수 있다. 오버레이는 Step이 진행되는 동안 계속 유지된다.
* background를 리스트로 지정하면 여러 배경이 순환하며 표시된다.
* 렌더링 순서는 background -> guide_video -> image -> text -> popup -> detection -> button -> Information Area 순서이다.

Button 구조
--------------
``` 
{
  "name": 버튼 이름,
  "bbox": 버튼의 (X, Y, W, H) 정의
  "angle": bbox의 중심을 기준으로 버튼의 회전 각도 정의(deg, CCW +), 정의하지 않으면 0
  "timer": 버튼을 누를 시간 정의(ms)
  "action": 이 항목이 존재할 경우 Step의 completion_events를 무시하고 정의된 동작 실행(RESTART, PREVIOUS, NEXT, RECORD, Step 번호나 이름)
  "press_up_image": 누르지 않았을 때 표시할 이미지(선택)
  "press_down_image": 눌렀을 때 표시할 이미지(선택)
  "press_down_sound": 버튼을 누를 때 재생할 효과음 파일의 위치(선택)
  "press_end_sound": 버튼을 누르고 정해진 시간을 다 채웠을 때 재생할 효과음(선택)
  "desired_work_hand": 작업에 사용해야 하는 손(Recent, Left, Right 중 선택, 정의하지 않으면 지정되지 않음)
  "completion_event": 버튼을 일정 시간 동안 눌렀을 때 달성시킬 조건 이름(Step의 completion_events에 정의되어 있어야 함)
  "handle_grasp": true로 설정할 경우 버튼 위에서 손이 일정 범위 이상 움직이면 타이머가 초기화됨(손 안정성 검사, 선택)
  "display_delay": ms 단위로 시간을 정의하면 해당 시간이 지난 뒤에 해당 버튼이 표시되고 사용할 수 있음
}
```
* angle 항목을 통해 버튼을 회전시킬 수 있다. 회전할 시 버튼 이미지와 클릭 영역 모두 같이 회전된다.
* action 항목에 사용 가능한 값:
  * `RESTART`: 시나리오를 처음부터 다시 시작
  * `PREVIOUS`: 이전 Step으로 이동
  * `NEXT`: 다음 Step으로 이동
  * `RECORD`: 버튼 클릭 기록을 남김(학습 기록용)
  * Step 번호 또는 이름: 해당 Step으로 이동

Image 구조
--------------
```
{
  "name": 이미지 이름(선택, Step 내에서 유일해야 함),
  "path": 표시할 이미지 파일 경로,
  "bbox": 이미지의 (X, Y, W, H) 정의,
  "angle": bbox 중심을 기준으로 이미지의 회전 각도(deg, CCW +, 선택, 기본값 0),
  "display_delay": Step 시작 후 표시를 지연시킬 시간(ms, 선택, 기본값 0),
  "visible": true/false로 초기 표시 여부를 지정(선택, 기본값 true)
}
```
* 이미지 위치와 크기는 World 좌표계[cm]로 작성하며, 시스템에서 Pixel 좌표계로 자동 변환된다.
* angle을 통해 Oriented Bounding Box 기반으로 이미지를 회전할 수 있다.
* display_delay를 사용하면 Step이 시작된 뒤 일정 시간 이후에 이미지를 띄울 수 있다.

Detection 구조
--------------
``` 
{
  "name": detection 이름
  "bbox": 인식 영역의 (X, Y, W, H) 정의
  "timer": 인식 시간 정의(ms)
  "class": YOLO 클래스 번호 혹은 이름
  "default_image": {"path": "~~", "bbox": []} 형태로 인식 범위에 사물이 없을 때 표시할 이미지(선택)
  "incorrect_image": {"path": "~~", "bbox": []} 형태로 인식 범위에 잘못된 사물이 있을 때 표시할 이미지(선택)
  "correct_image": {"path": "~~", "bbox": []} 형태로 인식 범위에 올바른 사물이 있을 때 표시할 이미지(선택)
  "completion_event": 인식되었을 때 달성시킬 조건 이름(Step의 completion_events에 이 이름이 정의되어 있어야 함)
}
```

Popup 구조
--------------
``` 
{
  "name": 팝업 이름,
  "bbox": 팝업의 (X, Y, W, H) 정의
  "timer": 팝업을 표시할 시간 정의(ms)
  "image": 팝업에 표시할 이미지
  "completion_event": 팝업이 닫힐 때 달성할 조건 이름(Step의 completion_events에 정의되어 있어야 함, 선택)
  "sound": 팝업이 표시될 때 재생할 사운드 파일 경로(선택)
}
```
팝업이 로드될 때 completion_event에 ```popup:팝업_이름``` 이름의 이벤트가 자동으로 생성되며, 이 이벤트를 달성시키면 팝업이 표시된다.
시스템 제약 상 각 팝업은 한 번만 열릴 수 있다.

Text 구조
--------------
```
{
  "name": 텍스트 이름,
  "text": 표시할 문자열,
  "position": [X, Y] 형태의 World 좌표계 위치 정의,
  "align": 정렬 방식(선택, 기본값 left, center 지정 시 position이 텍스트 중앙),
  "size": 텍스트 높이를 cm 단위로 정의(World 좌표계 기준),
  "color": [R, G, B] 혹은 "#RRGGBB" 형태의 색상 정의(선택)
}
```

* position은 텍스트의 기준 위치이며, World 좌표계[cm]로 작성한다. Y축 위치는 텍스트의 중심으로 고정되어 있으며, X축 위치는 align 설정에 따라 달라진다.
* align 값은 position에서 지정한 위치 값을 텍스트의 어느 위치로 설정할 지 정하는 옵션이다. (기본값: left)
  * `left`인 경우 position이 텍스트의 좌중간 지점이 된다. (왼쪽 정렬)
  * `center`를 지정하면 position이 텍스트의 중심 위치가 된다. (가운데 정렬)
  * `right`인 경우 position이 텍스트의 우중간 지점이 된다. (오른쪽 정렬)
* size는 텍스트 높이를 의미하며 World 좌표계[cm]로 작성한다. `\n`을 사용하여 여러 줄로 작성할 수 있다.
* color를 지정하지 않으면 흰색(255, 255, 255)이 사용되며, 범위를 벗어난 RGB 값이나 잘못된 형식은 로드 시 오류로 처리된다.

LPIPS 구조
--------------
LPIPS(Learned Perceptual Image Patch Similarity)는 카메라로 촬영한 작업 영역의 이미지를 레퍼런스 이미지와 비교하여 조립 상태를 자동으로 판정하는 기능이다. 지정된 영역을 주기적으로 촬영하고, LPIPS 거리가 threshold 미만이면 조건을 달성시킨다.
``` 
{
  "name": LPIPS 이름,
  "bbox": 비교 영역의 (X, Y, W, H) 정의,
  "reference": 레퍼런스 이미지 파일 경로,
  "threshold": LPIPS 거리 임계값(float, 이 값 미만이면 통과),
  "completion_event": 비교 통과 시 달성시킬 조건 이름(Step의 completion_events에 정의되어 있어야 함),
  "autocheck_interval": 자동 비교 주기(ms, 선택),
  "autocheck_hand_free_interval": 손이 비교 영역에서 벗어난 뒤 비교를 시작할 대기 시간(ms, 선택),
  "use_mask": 비교 시 프로젝터 영역을 블랙 마스크로 가릴지 여부(선택, 기본값 true),
  "mask_margin": 블랙 마스크의 확장 마진 [가로, 세로](cm, 선택, 기본값 [3, 3]),
  "use_alignment": 프로젝터-카메라 오정렬 보정 활성화 여부(선택, 기본값 true),
  "alignment_margin": 정렬 탐색을 위한 확장 마진 [가로, 세로](cm, 선택, 기본값 [1, 1]),
  "alignment_method": 정렬 방식(선택, 기본값 "multi_offset", 아래 설명 참고),
  "alignment_search_step": multi_offset 방식에서 오프셋 탐색 간격(pixel, 선택, 기본값 4),
  "alignment_blur_sigma": 비교 전 Gaussian Blur의 sigma 값(선택, 기본값 0.5, 0이면 비활성화)
}
```

* LPIPS가 로드될 때 completion_event에 ```lpips:LPIPS_이름``` 이름의 이벤트가 자동으로 생성되며, 이 이벤트를 달성시키면 LPIPS 비교가 트리거된다. (버튼을 통한 트리거 등에 사용)
* threshold 값은 0에 가까울수록 엄격하며, 일반적으로 0.05~0.10 사이의 값을 사용한다.
* autocheck_interval을 지정하면 해당 주기마다 자동으로 비교가 수행된다. autocheck_hand_free_interval을 지정하면 손이 비교 영역에서 벗어난 후 지정된 시간이 지나면 비교가 수행된다. 둘 다 지정하지 않으면 수동 트리거만 가능하다.
* use_mask가 true이면 비교 시 프로젝터가 투사하는 빛의 영향을 제거하기 위해 해당 영역에 블랙 마스크를 적용하여 카메라 프레임을 캡처한다.
* bbox와 mask_margin, alignment_margin은 모두 World 좌표계[cm]로 작성한다.

**오정렬 보정(Alignment) 설명:**

빔 프로젝터와 카메라의 관계가 미세하게 틀어질 경우 LPIPS 점수가 올라가 유사한 이미지임에도 통과하지 못할 수 있다. use_alignment를 활성화하면 alignment_margin만큼 확장된 영역을 캡처하여 오프셋을 보정한 뒤 비교한다.

* alignment_method에는 다음 세 가지 방식이 있다.
  * `multi_offset`(기본값): 확장된 크롭 영역 내에서 여러 오프셋 위치의 이미지를 생성하고, LPIPS 거리를 배치로 계산하여 최솟값을 사용한다. LPIPS 점수 자체를 직접 최적화하므로 가장 효과적이다.
  * `ecc`: 템플릿 매칭으로 초기 오프셋을 찾은 뒤, ECC(Enhanced Correlation Coefficient) 알고리즘으로 이동과 회전을 정밀 보정한다. 실패 시 template 방식으로 자동 전환된다.
  * `template`: 템플릿 매칭을 사용하여 최적 이동 오프셋만 보정한다.
* alignment_search_step은 multi_offset 방식에서 오프셋 탐색 격자의 간격(pixel)을 의미하며, 값이 작을수록 정밀하지만 계산량이 증가한다.
* alignment_blur_sigma는 비교 전 양쪽 이미지에 Gaussian Blur를 적용하여 미세한 픽셀 이동에 대한 LPIPS의 민감도를 낮춘다. 0으로 설정하면 비활성화된다. alignment_method와 독립적으로 동작한다.

Example
--------------
``` json
[
    {
        "name": "start",
        "background": ["backgrounds/0_START.png"],
        "guide_sound": "sounds/01_작업_지도를_시작합니다.mp3",
        "buttons": [
            {
                "name": "place_point",
                "bbox": [33.06, 34.25, 24.68, 8.89],
                "timer": 1000,
                "completion_event": "hover_hand",
                "press_down_image": "images/START_DOWN.png",
                "press_down_sound": "sounds/button_press_down.mp3"
            }
        ],
        "images": [
            {
                "name": "start_hint",
                "path": "images/START_HINT.png",
                "bbox": [25, 18, 20, 6],
                "angle": 0
            }
        ],
        "completion_events": ["hover_hand"]
    },
    {
        "name": "pick_roll",
        "background": ["backgrounds/1_1.png", "backgrounds/1_2.png"],
        "guide_sound": "sounds/02_반짝이는_곳에서_부품을_집으세요.mp3",
        "buttons": [
            {
                "name": "place_point",
                "bbox": [75.8, -10, 15, 15],
                "timer": 300,
                "completion_event": "hover_hand"
            },
            "import:scenarios/manager_buttons.json"
        ],
        "completion_events": ["hover_hand"]
    },
    {
        "name": "place_roll",
        "background": ["backgrounds/object_test.png"],
        "guide_sound": "sounds/03_부품을_그림에_맞게_놓으세요.mp3",
        "buttons": [
            "import:scenarios/manager_buttons.json"
        ],
        "detections": [
            {
                "name": "place_object",
                "bbox": [68, 26, 4, 3.5],
                "timer": 1500,
                "class": 5,
                "default_image": {"path": "images/part_6.png", "bbox": [68, 26, 4, 3.5]},
                "incorrect_image": {"path": "images/alarms/x.png", "bbox": [38, 28, 14, 14]},
                "completion_event": "place_object"
            }
        ],
        "texts": [
          {
            "name": "progress_text",
            "text": "진행 중",
            "position": [10, 5],
            "align": "left",
            "size": 3,
            "color": [255, 255, 255]
          }
        ],
        "completion_events": ["place_object"]
    },
    {
        "name": "assemble_roll",
        "background": ["backgrounds/3.png"],
        "guide_sound": "sounds/04_부품을_조립하세요.mp3",
        "guide_video": "videos/assemble_roll_crop.mkv",
        "guide_video_bbox": [16.66, 12.92, 57.49, 22],
        "buttons": [
            "import:scenarios/manager_buttons.json"
        ]
    },
    {
        "name": "restart",
        "background": ["backgrounds/RESTART.png"],
        "guide_sound": "sounds/05_완성입니다_다시_하시겠습니까.mp3",
        "buttons": [
            {
                "name": "yes",
                "bbox": [22.87, 34.25, 19.52, 8.89],
                "timer": 1000,
                "press_down_image": "images/RESTART_YES_DOWN.png",
                "press_down_sound": "sounds/button_press_down.mp3",
                "action": 1
            },
            {
                "name": "no",
                "bbox": [ 48.41, 34.25, 19.52, 8.89],
                "timer": 1000,
                "press_down_image": "images/RESTART_NO_DOWN.png",
                "press_down_sound": "sounds/button_press_down.mp3",
                "action": "NEXT"
            }
        ]
    },
    {
        "name": "end",
        "background": "backgrounds/END.png",
        "guide_sound": "sounds/06_수고하셨습니다.mp3",
        "completion_events": ["delay:2000"]
    }
]
```