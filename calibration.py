import cv2
import numpy as np

VIDEO_PATH = 'chessboard.mp4'
BOARD_SIZE = (7, 5)
SQUARE_SIZE = 1.0
FRAME_INTERVAL = 15
OUTPUT_FILE = 'calibration_result.npz'

objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

obj_points = []
img_points = []

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
found_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % FRAME_INTERVAL != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

    if found:
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        obj_points.append(objp)
        img_points.append(corners_refined)
        found_count += 1
        print(f'[{found_count}장 확보]')

cap.release()

print(f'\n총 {found_count}장으로 캘리브레이션 시작...')
h, w = gray.shape
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

print(f'RMS 오차: {ret:.4f}')
print(f'K:\n{K}')
np.savez(OUTPUT_FILE, K=K, dist=dist, rms=ret)
print(f'\n저장 완료: {OUTPUT_FILE}')