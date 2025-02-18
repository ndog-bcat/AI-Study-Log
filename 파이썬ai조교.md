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
- ![image](https://github.com/user-attachments/assets/9c30dfca-b717-4338-b35a-13507dc8f423)
- 이미지 조정 전 텍스트 추출
- ![image](https://github.com/user-attachments/assets/8d657809-19d3-4955-a25b-4ec40dbd5a2c)
- 이미지 조정 후 텍스트 추출
- ![image](https://github.com/user-attachments/assets/69d2aac0-feca-47c2-a2a7-a93a9c12c29e)
- 돌려보고 정말 기뻤다. 영어와 한글 모두 확연하게 인식률이 오른 것을 볼 수 있었다.
- 본 코드 파일
- 바뀐점 : 이진화 여부 변수 추가, 이미지 전처리 함수 파라미터 추가(이진화 여부), 텍스트 추출 함수 추가
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
from text_process import extract_text

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

apply_threshold = False # 이진화 여부 변수 (이진화를 원하면 True 아니면 False지만 경험상 안쓰는게 나을듯)

while True:
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    screenshot = image_processing(screenshot, apply_threshold) # 이미지 전처리
    text = extract_text(screenshot) # 텍스트 추출
    errors = analyze_code(text)  # 오류검사
    if errors:
        display_errors(errors)  # 수정안 이미지 반환
    time.sleep(5)  # 5초마다 체크
```
- 이미지 전처리 파일
- 바뀐점 : 글씨 높이 기반 이미지 크기 조정 추가, 이진화 여부 파라미터 추가(하지만 안쓰는게 인식률이 좋아 쓰지않을 것으로 예상됨), 노이즈제거는 챗지피티가 말한 가우시안블러(속도 빠름)보다 정확성을 위해 있던 걸 썼다. 그러나 그레이스케일변환을 먼저 해서 노이즈제거 함수도 그에 맞는 걸 써야해서 'Colored'가 빠진 걸 볼 수 있다. 그리고 height,width,_가 아닌 height,width로 받게 된 이유도 위에 테스트에서 사용한 컬러이미지는 색상정보가 있어 3차원정보를 받았지만 본 코드에서는 그레이이미지로 받기 때문에 gray.shape 안에는 height와 width만 들어가서 그렇다.
```
import cv2
import numpy as np
import pytesseract
# 이미지 전처리 함수
def image_processing(screenshot, apply_threshold):
    # 그레이스케일 변환
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)


    # 대비 향상 : 이진화 (배경에서 코드 추출 잘 되도록)
    if apply_threshold:
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        
    # 글자 높이 기반 이미지 크기 조정
    height, width = gray.shape
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    # 글자 크기 측정
    heights = [int(data['height'][i]) for i in range(len(data['text'])) if data['text'][i].strip() != '']
    average_height = sum(heights) / len(heights)

    # 이미지 크기 조정 비율
    scale_factor = 30 / average_height
    new_size = (int(width*scale_factor), int(height*scale_factor))

    # 이미지 크기 변경
    gray = cv2.resize(gray, new_size)

    return gray
```
- 텍스트 추출 파일(새로 생김)
- https://m.blog.naver.com/johnsmithbrainseven/222242853850 (장풍님 블로그 글)
- 여기에 oem과 psm의 설명이 되어있다. oem에서 legacy엔진 쓰냐 lstm엔진쓰냐 이런게 뭔가 하니 언어학습에 쓰인 데이터가 legacy엔진을 위해 학습된 건지 lstm엔진을 위해 학습된 건지가 정해져있어서 엔진모드를 정해주는 거였다. 나는 tessdata를 다운받았고 tessdata는 둘 다 되는 모양이다.
```
import pytesseract

def extract_text(img):
    # OCR 설정: psm6(한글인식.....텍스트의 균일한 단일 블록을 가정함)-코드 블록 적합
    custom_config = r"--oem 3 --psm 6"

    # 텍스트 추출
    extracted_text = pytesseract.image_to_string(img, lang="eng+kor", config=custom_config)

    return extracted_text
```
- 이미지전처리함수와 텍스트추출함수의 테스트 코드파일
```
import pyautogui
import pytesseract
import cv2
import numpy as np
from python_error_finder.image_process import image_processing
from python_error_finder.text_process import extract_text

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

apply_threshold = False

screenshot = pyautogui.screenshot(region=(x, y, width, height))
screenshot = np.array(screenshot)
screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
screenshot = image_processing(screenshot, apply_threshold)
text = extract_text(screenshot)
print(text)
```
