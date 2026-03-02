import cv2
import numpy as np
import math
import joblib

from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks

print("Loading models...")
face_model     = get_face_detector()
landmark_model = get_landmark_model()
clf            = joblib.load('models/face_spoofing.pkl')
print("All models loaded!")

font = cv2.FONT_HERSHEY_SIMPLEX

# ── Eye tracker constants ──────────────────────────────
left_eye  = [36, 37, 38, 39, 40, 41]
right_eye = [42, 43, 44, 45, 46, 47]
kernel    = np.ones((9, 9), np.uint8)

def nothing(x):
    pass

def eye_on_mask(mask, side, shape):
    points = np.array([shape[i] for i in side], dtype=np.int32)
    mask   = cv2.fillConvexPoly(mask, points, 255)   # ✅ reassign mask
    l = points[0][0]
    t = (points[1][1] + points[2][1]) // 2
    r = points[3][0]
    b = (points[4][1] + points[5][1]) // 2
    return mask, [l, t, r, b]

def find_eyeball_position(ep, cx, cy):
    try:
        xr = (ep[0] - cx) / (cx - ep[2])
        yr = (cy - ep[1]) / (ep[3] - cy)
        if xr > 3:    return 1   # looking left
        if xr < 0.33: return 2   # looking right
        if yr < 0.33: return 3   # looking up
        return 0
    except:
        return 0

def contouring(thresh, mid, img, ep, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M   = cv2.moments(cnt)
        cx  = int(M['m10'] / M['m00'])
        cy  = int(M['m01'] / M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        return find_eyeball_position(ep, cx, cy)
    except:
        return 0

def process_thresh(thresh):
    thresh = cv2.erode(thresh,  None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 3)
    return cv2.bitwise_not(thresh)

# ── Head pose constants ────────────────────────────────
model_points = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0,  -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0)
])

# ── Mouth constants ────────────────────────────────────
outer_points = [[49,59],[50,58],[51,57],[52,56],[53,55]]
inner_points = [[61,67],[62,66],[63,65]]

# ── Face spoofing helper ───────────────────────────────
def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)

# ── MAIN ───────────────────────────────────────────────
def run_proctoring(video_path=0):
    cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Waiting for camera...")
    img = None
    for _ in range(30):
        ret, img = cap.read()
        if ret and img is not None:
            break
    if img is None:
        print("ERROR: Cannot open camera!")
        return

    print("Camera ready!")
    h, w = img.shape[:2]

    camera_matrix = np.array([
        [w, 0, w/2],
        [0, w, h/2],
        [0, 0,   1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    cv2.namedWindow("Eye Threshold")
    cv2.createTrackbar("threshold", "Eye Threshold", 75, 255, nothing)

    d_outer          = [0] * 5
    d_inner          = [0] * 3
    mouth_calibrated = False

    # track previous state for terminal print (avoid spam)
    prev_eye   = ""
    prev_head  = ""
    prev_mouth = ""
    prev_spoof = ""

    print("=" * 50)
    print("PROCTORING STARTED!")
    print("Press 'c' = calibrate mouth (keep mouth closed)")
    print("Press 'q' = quit")
    print("=" * 50)

    thresh_display = np.zeros((h, w), np.uint8)

    while True:
        ret, img = cap.read()
        if not ret or img is None:
            continue

        output = img.copy()
        faces  = find_faces(img, face_model)

        if len(faces) == 0:
            cv2.putText(output, "No face detected", (10, 40), font, 1, (0, 0, 255), 2)
            cv2.imshow("Proctoring System", output)
            cv2.imshow("Eye Threshold", thresh_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if len(faces) > 1:
            msg = f"WARNING: {len(faces)} faces detected!"
            cv2.putText(output, msg, (10, 40), font, 0.8, (0, 0, 255), 2)
            if prev_head != msg:
                print(msg)
                prev_head = msg

        for face in faces:
            shape    = detect_marks(img, landmark_model, face)
            x, y, x1, y1 = face

            # ══ 1. EYE TRACKING ══════════════════════════
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask, ep_left  = eye_on_mask(mask, left_eye,  shape)
            mask, ep_right = eye_on_mask(mask, right_eye, shape)
            mask_dilated   = cv2.dilate(mask, kernel, 5)

            eyes = cv2.bitwise_and(img, img, mask=mask_dilated)
            eyes[(eyes == [0,0,0]).all(axis=2)] = [255, 255, 255]

            mid       = int((shape[42][0] + shape[39][0]) // 2)
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            tval      = cv2.getTrackbarPos('threshold', 'Eye Threshold')
            _, thresh = cv2.threshold(eyes_gray, tval, 255, cv2.THRESH_BINARY)
            thresh    = process_thresh(thresh)
            thresh_display = thresh.copy()

            pos_l = contouring(thresh[:, 0:mid], mid, output, ep_left,  False)
            pos_r = contouring(thresh[:, mid:],  mid, output, ep_right, True)

            if pos_l == pos_r and pos_l != 0:
                if pos_l == 1: eye_text = "Looking Left"
                if pos_l == 2: eye_text = "Looking Right"
                if pos_l == 3: eye_text = "Looking Up"
            else:
                eye_text = "Eyes: OK"

            cv2.putText(output, eye_text, (10, 30), font, 0.7, (0, 255, 255), 2)

            # ✅ Print to terminal only when state changes
            if eye_text != "Eyes: OK" and eye_text != prev_eye:
                print(f"[EYE]   {eye_text}")
                prev_eye = eye_text
            elif eye_text == "Eyes: OK":
                prev_eye = ""

            # ══ 2. HEAD POSE ═════════════════════════════
            try:
                ip = np.array([
                    shape[30], shape[8],  shape[36],
                    shape[45], shape[48], shape[54]
                ], dtype="double")
                _, rvec, tvec = cv2.solvePnP(model_points, ip, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
                nose_end, _   = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rvec, tvec, camera_matrix, dist_coeffs)
                p1 = (int(ip[0][0]), int(ip[0][1]))
                p2 = (int(nose_end[0][0][0]), int(nose_end[0][0][1]))
                cv2.line(output, p1, p2, (0, 255, 255), 2)

                try:    ang1 = int(math.degrees(math.atan((p2[1]-p1[1])/(p2[0]-p1[0]))))
                except: ang1 = 90

                if   ang1 >= 48:  head_text = "Head Down"
                elif ang1 <= -48: head_text = "Head Up"
                else:             head_text = "Head: OK"

                cv2.putText(output, head_text, (10, 60), font, 0.7, (255, 255, 0), 2)

                if head_text != "Head: OK" and head_text != prev_head:
                    print(f"[HEAD]  {head_text}")
                    prev_head = head_text
                elif head_text == "Head: OK":
                    prev_head = ""
            except:
                cv2.putText(output, "Head: OK", (10, 60), font, 0.7, (255, 255, 0), 2)

            # ══ 3. MOUTH ═════════════════════════════════
            if not mouth_calibrated:
                cv2.putText(output, "Mouth: Press 'c' to calibrate", (10, 90), font, 0.6, (0, 200, 255), 2)
            else:
                cnt_o = sum(1 for i,(p1,p2) in enumerate(outer_points) if d_outer[i]+3 < shape[p2][1]-shape[p1][1])
                cnt_i = sum(1 for i,(p1,p2) in enumerate(inner_points) if d_inner[i]+2 < shape[p2][1]-shape[p1][1])
                if cnt_o > 3 and cnt_i > 2:
                    mouth_text = "Mouth OPEN!"
                    cv2.putText(output, mouth_text, (10, 90), font, 0.7, (0, 0, 255), 2)
                    if mouth_text != prev_mouth:
                        print(f"[MOUTH] {mouth_text}")
                        prev_mouth = mouth_text
                else:
                    cv2.putText(output, "Mouth: Closed", (10, 90), font, 0.7, (0, 255, 0), 2)
                    prev_mouth = ""

            # ══ 4. FACE SPOOFING ═════════════════════════
            roi = img[max(0,y):y1, max(0,x):x1]
            if roi.size > 0:
                try:
                    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
                    luv   = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)
                    h1    = calc_hist(ycrcb)
                    h2    = calc_hist(luv)
                    fv    = np.array([
                        h1[0].mean(), h1[1].mean(), h1[2].mean(),
                        h2[0].mean(), h2[1].mean(), h2[2].mean()
                    ]).reshape(1, -1)
                    prob = clf.predict_proba(fv)[0][1]

                    if prob >= 0.7:
                        spoof_text = "FAKE FACE"
                        cv2.rectangle(output, (x,y), (x1,y1), (0,0,255), 2)
                        cv2.putText(output, spoof_text,    (x, y-8),  font, 0.7, (0,0,255), 2)
                        cv2.putText(output, "Spoof: FAKE", (10, 120), font, 0.7, (0,0,255), 2)
                        if spoof_text != prev_spoof:
                            print(f"[SPOOF] WARNING - FAKE FACE DETECTED!")
                            prev_spoof = spoof_text
                    else:
                        cv2.rectangle(output, (x,y), (x1,y1), (0,255,0), 2)
                        cv2.putText(output, "Real",        (x, y-8),  font, 0.7, (0,255,0), 2)
                        cv2.putText(output, "Spoof: Real", (10, 120), font, 0.7, (0,255,0), 2)
                        prev_spoof = ""
                except:
                    cv2.rectangle(output, (x,y), (x1,y1), (255,255,0), 2)

        cv2.putText(output, "c=calibrate mouth | q=quit", (10, h-10), font, 0.5, (180,180,180), 1)
        cv2.imshow("Proctoring System", output)
        cv2.imshow("Eye Threshold", thresh_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("Calibrating mouth... keep mouth CLOSED")
            d_outer = [0]*5
            d_inner = [0]*3
            count   = 0
            for _ in range(100):
                ret2, img2 = cap.read()
                if ret2 and img2 is not None:
                    faces2 = find_faces(img2, face_model)
                    for face2 in faces2:
                        shape2 = detect_marks(img2, landmark_model, face2)
                        for i,(p1,p2) in enumerate(outer_points):
                            d_outer[i] += shape2[p2][1] - shape2[p1][1]
                        for i,(p1,p2) in enumerate(inner_points):
                            d_inner[i] += shape2[p2][1] - shape2[p1][1]
                        count += 1
            if count > 0:
                d_outer[:] = [v/count for v in d_outer]
                d_inner[:] = [v/count for v in d_inner]
                mouth_calibrated = True
                print("Mouth calibrated!")
            else:
                print("Calibration failed - no face detected!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    src = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    run_proctoring(src)