import cv2
import numpy as np
from PIL import Image  # pip install Pillow

# ========== 설정 ==========
VIDEO_PATH = 'chessboard.mp4'
CALIB_FILE = 'calibration_result.npz'
GIF_PATH = 'overlay.gif'       # 사용할 gif 파일
BOARD_SIZE = (7, 5)            # 내부 코너 수 (가로, 세로)
SQUARE_SIZE = 1.0
OUTPUT_VIDEO = 'ar_result.mp4' # 결과 저장 (None이면 저장 안 함)
# ==========================


def load_gif_frames(gif_path):
    """GIF 파일에서 모든 프레임을 BGR + alpha로 읽기"""
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = gif.convert('RGBA')
            frames.append(np.array(frame))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames


def overlay_gif_on_board(frame, gif_bgra, corners, board_size, K, dist, rvec, tvec):
    h_gif, w_gif = gif_bgra.shape[:2]
    
    # 체스보드 중앙 위치
    cx = (board_size[0] - 1) / 2.0
    cy = (board_size[1] - 1) / 2.0
    height = 4.0  # 높이 (값 키우면 더 크게 서 있음)
    width = height * w_gif / h_gif  # 비율 유지

    # 수직으로 세운 평면의 3D 좌표 (체스보드 위로 솟아오름)
    obj_pts = np.float32([
        [cx - width/2, cy, -height],  # 좌상단
        [cx + width/2, cy, -height],  # 우상단
        [cx + width/2, cy,  0],       # 우하단
        [cx - width/2, cy,  0],       # 좌하단
    ])

    # 3D → 2D 투영
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    img_pts = img_pts.reshape(-1, 2)

    # GIF 4꼭짓점 (src)
    src_pts = np.float32([
        [0, 0],
        [w_gif, 0],
        [w_gif, h_gif],
        [0, h_gif]
    ])

    H, _ = cv2.findHomography(src_pts, img_pts)

    h_frame, w_frame = frame.shape[:2]
    warped = cv2.warpPerspective(gif_bgra, H, (w_frame, h_frame))

    alpha = warped[:, :, 3:4] / 255.0
    gif_bgr = cv2.cvtColor(warped[:, :, :3], cv2.COLOR_RGB2BGR)

    result = frame.copy().astype(np.float32)
    result = result * (1 - alpha) + gif_bgr.astype(np.float32) * alpha
    return result.astype(np.uint8)


def draw_axes(frame, K, dist, rvec, tvec, square_size):
    """카메라 pose 시각화 - 좌표축 그리기"""
    axis_len = square_size * 3
    axis_pts = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, -axis_len]
    ])
    imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, dist)
    imgpts = imgpts.astype(int)

    origin = tuple(imgpts[0].ravel())
    cv2.arrowedLine(frame, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 3, tipLength=0.2)   # X: 빨강
    cv2.arrowedLine(frame, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 3, tipLength=0.2)   # Y: 초록
    cv2.arrowedLine(frame, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 3, tipLength=0.2)   # Z: 파랑
    cv2.putText(frame, 'X', tuple(imgpts[1].ravel()), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(frame, 'Y', tuple(imgpts[2].ravel()), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, 'Z', tuple(imgpts[3].ravel()), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    return frame


def run():
    # 캘리브레이션 결과 로드
    data = np.load(CALIB_FILE)
    K = data['K']
    dist = data['dist']
    print(f"[INFO] 캘리브레이션 결과 로드 완료")

    # GIF 프레임 로드
    gif_frames = load_gif_frames(GIF_PATH)
    print(f"[INFO] GIF 프레임 수: {len(gif_frames)}")

    # 3D 오브젝트 포인트
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] 동영상을 열 수 없어요: {VIDEO_PATH}")
        return

    # 결과 저장 설정
    writer = None
    if OUTPUT_VIDEO:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    gif_idx = 0
    print("[INFO] AR 실행 중... 'q'를 누르면 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

        if found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            # Pose estimation (PnP)
            ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners_refined, K, dist)

            if ret_pnp:
                # GIF 오버레이
                gif_bgra = gif_frames[gif_idx % len(gif_frames)]
                frame = overlay_gif_on_board(frame, gif_bgra, corners_refined, BOARD_SIZE, K, dist, rvec, tvec)

                gif_idx += 1  # 다음 GIF 프레임으로

        cv2.imshow('AR - GIF on Chessboard', frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"[INFO] 결과 저장 완료: {OUTPUT_VIDEO}")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()