# openCV와 diffusers 라이브러리를 숙달이 목표인 프로젝트입니다.

# 파이썬ai조교의 목적
- 우리 학교의 파이썬 문법을 배우는 과목 기초인공지능프로그래밍에 도움이 될 프로그램
- 실시간 화면을 입력받아 텍스트 처리 후 문법적 오류를 발견하고 피드백 해줄거임
- 추후에 gui가 생기게 된다면 따로 실행버튼을 만들어 사용자가 필요시 화면 입력을 하게 할 것임
- 또 사용자가 화면 캡처 영역을 설정할 수 있게 할 것임. 현재는 컴파일러를 전체화면으로 하는 것으로 상정하고 전체화면 캡처 예정

# 파이썬ai조교의 기능 과정 구상
1. 프로그램 실행
2. n초 주기로 실시간 화면 입력 (추후 gui가 생겼을 때 사용자가 직접 입력제어)
3. 이미지 전처리
4. 텍스트 추출
5. 문법 오류 검사
6. 오류 지점 수정안 첨가 후 이미지 출력

# 사용하게 된 라이브러리 및 기능
- openCV : 이미지 처리
- pyautogui : 화면 캡처
- pytesseract : 텍스트 추출
- ast : 문법 오류 검사

# 2025-01-27
- 몸체 파일
```
# 실시간 화면 캡처를 통해 기초 문법적 오류를 수정해주는 프로그램
import time
import pyautogui
import pytesseract
import cv2
import numpy as np
import keyboard
from image_process import image_processing
from error_process import analyze_code, display_errors

def on_press(key):
    if key.name == 'esc':
        print("program end")
        return False #프로그램 종료 함수 (사용자가 esc키를 누르면 종료)
    
keyboard.on_press(on_press) # 키보드 이벤트 리스너 설정

answer = 'no'

while answer == 'no':
    get_region = np.array(pyautogui.screenshot())
    get_region = cv2.cvtColor(get_region, cv2.COLOR_RGB2BGR)
    x, y, width, height = cv2.selectROI(windowName='Drag mouse to select region. When youre done, press enter', img=get_region)
    selected_region = get_region[y:y+height, x:x+width] # 선택영역 이미지 잘라내기
    
    cv2.imshow('show you the region for 3 seconds', selected_region)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
    answer = input('Are you sure that this region is you wanted?(answer is yes or no)')
    # 앞으로 지켜볼 영역 지정

while True:
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    screenshot = image_processing(screenshot) # 이미지 전처리
    text = pytesseract.image_to_string(screenshot)
    errors = analyze_code(text)  # 텍스트 추출 및 오류검사
    if errors:
        display_errors(errors)  # 수정안 이미지 반환
    time.sleep(5)  # 5초마다 체크
```
- 이미지 전처리 파일
```
import cv2
# 이미지 전처리 함수
def image_processing(screenshot):
    # 노이즈 제거
    denoised = cv2.fastNlMeansDenoisingColored(screenshot, None, 10, 10, 7, 21)

    # 그레이스케일 변환
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    # 대비 향상 (배경에서 코드 추출 잘 되도록)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary
```
- 문법 오류 수정 및 수정안 파일 (아직 안 만들었음)
```
# 오류 검사
def analyze_code():
    return
# 수정안 반환
def display_errors(error):
    return
```

