# Animal Crossing AR

체스보드 위에 동물의 숲 캐릭터 GIF를 올려놓는 AR 프로젝트입니다.

## 데모

![demo](result.gif)

## 주요 기능

- 체스보드 동영상으로 카메라 캘리브레이션
- PnP 알고리즘을 이용한 실시간 카메라 자세 추정
- 캐릭터 GIF가 체스보드 위에 서 있는 것처럼 표시

## 필요 라이브러리 설치

```bash
pip install opencv-python numpy Pillow
```

## 실행 방법

### 1단계: 카메라 캘리브레이션

```bash
python calibration.py
```

`chessboard.mp4`를 읽어서 캘리브레이션 결과를 `calibration_result.npz`에 저장합니다.

### 2단계: AR 실행

원하는 GIF 파일을 `overlay.gif`로 이름 바꿔서 같은 폴더에 넣고 실행합니다.

```bash
python ar_gif.py
```

결과 영상이 `ar_result.mp4`로 저장됩니다.

## 파일 구조

```
.
├── calibration.py           # 카메라 캘리브레이션
├── ar_gif.py                # AR GIF 오버레이
├── chessboard.mp4           # 체스보드 입력 영상
├── overlay.gif              # 오버레이할 GIF
├── calibration_result.npz   # 캘리브레이션 결과 (자동 생성)
└── ar_result.mp4            # AR 결과 영상 (자동 생성)
```

## 동작 원리

1. **캘리브레이션** — 체스보드 영상에서 코너를 검출해 카메라 내부 파라미터와 왜곡 계수를 계산
2. **자세 추정** — 매 프레임마다 `cv2.solvePnP()`로 체스보드 기준 카메라의 회전/이동 벡터 추정
3. **GIF 오버레이** — 추정된 자세를 바탕으로 수직 평면을 투영하고, GIF 프레임을 알파 블렌딩으로 합성
