import cv2
import numpy as np

# --- 설정 ---
video_path = "video.mp4"
checkerboard_size = (10, 7)
square_size = 0.024  # 2.4cm

camera_matrix = np.array([
    [893.7861, 0, 534.6597],
    [0, 895.2723, 957.1658],
    [0, 0, 1]
])
dist_coeffs = np.array([0.0266, -0.0151, -0.0009, -0.0004, -0.0475])

# 체커보드 3D 점 생성
objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 정사면체 정의 (기존 위치에서 → 오른쪽 위로 이동)
a = 0.08
h = np.sqrt(2 / 3) * a
offset = np.array([0.12, 0.06, 0])
tetra_points = np.float32([
    [0, 0, 0],
    [a, 0, 0],
    [a/2, np.sqrt(3)*a/2, 0],
    [a/2, np.sqrt(3)*a/6, -h]
]) + offset

tetra_edges = [
    (0, 1), (1, 2), (2, 0),
    (0, 3), (1, 3), (2, 3)
]
line_colors = [
    (0, 255, 255),  # 노랑
    (0, 0, 255),    # 빨강
    (255, 0, 255),  # 자홍
    (255, 255, 0),  # 하늘
    (0, 255, 0),    # 연두
    (255, 0, 0)     # 파랑
]

# 고정된 좌표계 축 (왼쪽 아래 원점)
axis_length = 0.05
axis_origin = np.float32([
    [0, 0, 0],
    [axis_length, 0, 0],
    [0, axis_length, 0],
    [0, 0, -axis_length]
])

# --- 영상 열기 ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ 영상 열기 실패")
    exit()

# 🔴 출력용 VideoWriter 추가
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter("ar_output.mp4", fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret_cb:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)

        if ret_pnp:
            # 정사면체
            imgpts, _ = cv2.projectPoints(tetra_points, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = imgpts.reshape(-1, 2).astype(int)
            for i, (start, end) in enumerate(tetra_edges):
                cv2.line(frame, tuple(imgpts[start]), tuple(imgpts[end]), line_colors[i], 2)

            # 좌표축
            axis_proj, _ = cv2.projectPoints(axis_origin, rvec, tvec, camera_matrix, dist_coeffs)
            axis_proj = axis_proj.reshape(-1, 2).astype(int)
            cv2.line(frame, tuple(axis_proj[0]), tuple(axis_proj[1]), (0, 0, 255), 2)
            cv2.line(frame, tuple(axis_proj[0]), tuple(axis_proj[2]), (0, 255, 0), 2)
            cv2.line(frame, tuple(axis_proj[0]), tuple(axis_proj[3]), (255, 0, 0), 2)
            cv2.putText(frame, 'X', tuple(axis_proj[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Y', tuple(axis_proj[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Z', tuple(axis_proj[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

    resized = cv2.resize(frame, (540, 960))
    cv2.imshow("AR Final Fix (Tetrahedron + Axis)", resized)

    # 🔴 프레임 저장
    video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
