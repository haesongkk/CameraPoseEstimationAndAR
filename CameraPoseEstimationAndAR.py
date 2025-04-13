import cv2
import numpy as np

# --- 설정 ---
video_path = "video.mp4"
checkerboard_size = (10, 7)
square_size = 0.024  # 2.4cm

# 카메라 내부 파라미터
camera_matrix = np.array([
    [893.7861, 0, 534.6597],
    [0, 895.2723, 957.1658],
    [0, 0, 1]
])
dist_coeffs = np.array([0.0266, -0.0151, -0.0009, -0.0004, -0.0475])

# 체커보드 3D 점 생성
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# --- 영상 열기 및 저장 설정 ---
cap = cv2.VideoCapture(video_path)
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('output_with_ar.mp4', fourcc, fps, (output_width, output_height))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# --- 영상 프레임 반복 ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret_cb:
        # 코너 보정
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        # pose 추정
        ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)

        if ret_pnp:
            # 텍스트 평면 정의 (10x10cm 크기의 정사각형)
            text_plane_3d = np.float32([
                [-0.05, -0.05, 0],
                [ 0.05, -0.05, 0],
                [ 0.05,  0.05, 0],
                [-0.05,  0.05, 0]
            ])
            text_plane_2d, _ = cv2.projectPoints(text_plane_3d, rvec, tvec, camera_matrix, dist_coeffs)
            pts_2d = text_plane_2d.reshape(-1, 2).astype(int)
            center = np.mean(pts_2d, axis=0).astype(int)

            # 디버그 출력
            print(f"[DEBUG] Frame AR center: {center.tolist()}")

            # AR 텍스트 'A' 출력
            cv2.putText(
                frame,
                'A',
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 255, 255), 4, cv2.LINE_AA
            )

    # 실시간 창 출력 (축소본)
    resized_frame = cv2.resize(frame, (540, 960))
    cv2.imshow("AR Video", resized_frame)

    # 영상 파일로 저장
    video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 종료 처리 ---
cap.release()
video_writer.release()
cv2.destroyAllWindows()
