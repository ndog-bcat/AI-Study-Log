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
- 설명
keyboard라이브러리로 

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
- 오늘 구현 기능 테스트 파일
```
from python_error_finder import image_process
import time
import pyautogui
import cv2
import numpy as np
import pytesseract

answer = 'no'

while answer == 'no':
    get_region = np.array(pyautogui.screenshot())
    get_region = cv2.cvtColor(get_region, cv2.COLOR_RGB2BGR)
    x, y, width, height = cv2.selectROI(windowName='Drag mouse to select region. When you done, push enter', img=get_region)
    cv2.destroyAllWindows()
    selected_region = get_region[y:y+height, x:x+width] # 선택영역 이미지 잘라내기
    
    cv2.imshow('selected region', selected_region)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
    answer = input('Is this region is you wanted?(answer is yes or no)')
    # 앞으로 지켜볼 영역 지정

screenshot = pyautogui.screenshot(region=(x, y, width, height))
screenshot = np.array(screenshot)
screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
screenshot = image_process.image_processing(screenshot) # 이미지 전처리

text = pytesseract.image_to_string(screenshot, lang = 'kor')
print(text)
```
#수정할 점
- 텍스트 추출 함수 인식도 처참함 (매우 중요, 이미지 전처리 보완 필요)
- 다언어 텍스트 추출 시 어려움이 있을 듯 하여 수정 필요
- 마우스 드래그 후 별도의 입력 없이 바로 캡처기능으로 이어지는 것이 좋을 듯 하다.
- 안내 텍스트 출력에 있어 한글 제한. 수정 필요

# 2025-02-18
- 일단 이미지 전처리도 중요하지만 ocr에서 글자높이가 중요하다는 정보를 구글에서 읽었다. 그래서 일단 이미지에서 글자 높이를 추출하고 난 후 cv2.resize로 글자를 넉넉하게 30픽셀로 만들어서 텍스트 추출을 해보도록 하겠다.
- 글자 크기 뽑는 테스트 파일
```
#글자 크기 잡기
import pytesseract
from PIL import Image
import os

desktop_path = r'C:\Users\admin\Desktop'
image_name = 'vscode캡처.png'
image_path = os.path.join(desktop_path, image_name)
image = Image.open(image_path)
data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

# 글자 크기 측정
heights = [int(data['height'][i]) for i in range(len(data['text'])) if data['text'][i].strip() != '']
average_height = sum(heights) // len(heights)

print(average_height)
```
- 이렇게 해서 글자크기를 뽑은 다음 scale_factor = 30/average_height로 정한 다음 cv2.resize를 이용해서 글자크기를 30픽셀이 되는 이미지 크기로 만들어서 텍스트 추출을 해야겠다.
- 글자 크기 조정 후 텍스트 추출 테스트 파일
```
import cv2
import pytesseract
import numpy as np
import pyautogui

# PyTesseract의 Tesseract 경로 설정 (Windows 환경이라면 필요)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(img, apply_threshold):
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 노이즈 제거 (GaussianBlur 적용)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 이진화 적용 (선택적)
    if apply_threshold:
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    return gray

def extract_text(img, apply_threshold):
    # 이미지 전처리
    processed_img = preprocess_image(img, apply_threshold)

    # OCR 설정: --psm 6은 코드 블록에 적합
    custom_config = r"--oem 3 --psm 6"

    # 텍스트 추출
    extracted_text = pytesseract.image_to_string(processed_img, lang="eng+kor", config=custom_config)

    return extracted_text

apply_threshold = False #이진화 여부

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

# 테스트할 이미지 경로
screenshot = pyautogui.screenshot(region=(x, y, width, height))
screenshot = np.array(screenshot)
screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
height, width, _ = screenshot.shape
data = pytesseract.image_to_data(preprocess_image(screenshot, apply_threshold), output_type=pytesseract.Output.DICT)

# 글자 크기 측정
heights = [int(data['height'][i]) for i in range(len(data['text'])) if data['text'][i].strip() != '']
average_height = sum(heights) / len(heights)

# 이미지 크기 조정 비율
scale_factor = 30 / average_height
new_size = (int(width*scale_factor), int(height*scale_factor))

# 이미지 크기 변경
screenshot = cv2.resize(screenshot, new_size)

# 텍스트 추출 실행
text_result = extract_text(screenshot, apply_threshold)
print(text_result)
```
- 참고로 이미지 전처리 부분은 챗지피티의 도움을 좀 받아서 샘플 코드를 받아 넣었다. 이 샘플 코드와 내 기존 전처리 파일 코드를 비교해서 개선해서 내 코드를 고쳐야겠다. 이미지 전처리 중간의 oem과 psm이 뭔지 알아보았는데 설명은 밑에 넣어보겠다.
- 텍스트 추출 할 이미지
[![image](https://github.com/user-attachments/assets/9c30dfca-b717-4338-b35a-13507dc8f423)]
- 이미지 조정 전 텍스트 추출
[![image](https://github.com/user-attachments/assets/6664a5e2-a4a5-48e1-b4b1-029faa2b4e4d)]
- 이미지 조정 후 텍스트 추출
[![image](https://github.com/user-attachments/assets/69d2aac0-feca-47c2-a2a7-a93a9c12c29e)]
