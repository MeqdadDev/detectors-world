"""Regenerate the landmark maps and id tables used by the documentation.

The ids and connections come from MediaPipe, so the maps always match the models:

    python tools/generate_landmark_maps.py path/to/frontal-face-photo.png
"""

import sys
from pathlib import Path

import cv2 as cv
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmark

from detectors_world import FaceMesh

ROOT = Path(__file__).resolve().parent.parent
MAPS = ROOT / "docs" / "assets" / "maps"
SNIPPETS = ROOT / "docs" / ".snippets"
FACE = vision.FaceLandmarksConnections

INK = "#0d3934"
MUTED = "#5f7471"
SOFT = "#dfe9e7"
RIGHT_SIDE = "#d9622b"
LEFT_SIDE = "#2a7de1"
FINGERS = {"Thumb": "#d9622b", "Index": "#2a7de1", "Middle": "#0e8f7e", "Ring": "#8a4fc7", "Pinky": "#c43d7a"}

HAND_POINTS = {
    0: (300, 590), 1: (216, 540), 2: (150, 478), 3: (104, 420), 4: (62, 366),
    5: (214, 330), 6: (190, 238), 7: (174, 170), 8: (160, 104),
    9: (290, 312), 10: (286, 206), 11: (282, 132), 12: (280, 60),
    13: (364, 326), 14: (380, 228), 15: (392, 162), 16: (402, 100),
    17: (434, 362), 18: (476, 288), 19: (504, 238), 20: (530, 186),
}
POSE_BODY = {
    11: (370, 190), 12: (190, 190), 13: (450, 290), 14: (110, 290), 15: (500, 385), 16: (60, 385),
    17: (532, 442), 18: (28, 442), 19: (498, 462), 20: (62, 462), 21: (456, 430), 22: (104, 430),
    23: (335, 450), 24: (225, 450), 25: (355, 620), 26: (205, 620), 27: (365, 780), 28: (195, 780),
    29: (340, 836), 30: (220, 836), 31: (420, 846), 32: (140, 846),
}
POSE_FACE = {
    0: (730, 262), 1: (760, 200), 2: (795, 192), 3: (828, 200), 4: (700, 200), 5: (665, 192), 6: (632, 200),
    7: (858, 246), 8: (602, 246), 9: (778, 326), 10: (682, 326),
}
FACE_POINTS = {
    0: ("Right eye", 170, 210), 1: ("Left eye", 310, 210), 2: ("Nose tip", 240, 285),
    3: ("Mouth center", 240, 365), 4: ("Right ear tragion", 62, 245), 5: ("Left ear tragion", 418, 245),
}
MESH_LABELS = {
    10: "Forehead", 152: "Chin", 1: "Nose tip", 33: "Right eye outer", 133: "Right eye inner",
    362: "Left eye inner", 263: "Left eye outer", 468: "Right iris center", 473: "Left iris center",
    61: "Mouth right corner", 291: "Mouth left corner", 13: "Upper lip", 14: "Lower lip",
}
MESH_REGIONS = {
    "Face outline": (FACE.FACE_LANDMARKS_FACE_OVAL, "#0e8f7e"),
    "Lips": (FACE.FACE_LANDMARKS_LIPS, "#c43d7a"),
    "Left eye": (FACE.FACE_LANDMARKS_LEFT_EYE, "#2a7de1"),
    "Right eye": (FACE.FACE_LANDMARKS_RIGHT_EYE, "#2a7de1"),
    "Left eyebrow": (FACE.FACE_LANDMARKS_LEFT_EYEBROW, "#d9622b"),
    "Right eyebrow": (FACE.FACE_LANDMARKS_RIGHT_EYEBROW, "#d9622b"),
    "Left iris": (FACE.FACE_LANDMARKS_LEFT_IRIS, "#8a4fc7"),
    "Right iris": (FACE.FACE_LANDMARKS_RIGHT_IRIS, "#8a4fc7"),
}


def svg(width, height, title, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{title}" '
        f'font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">\n<title>{title}</title>\n'
        f'<rect width="{width}" height="{height}" rx="14" fill="#ffffff"/>\n' + "\n".join(body) + "\n</svg>\n"
    )


def line(a, b, color, width):
    return (
        f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round"/>'
    )


def text(x, y, content, size=15, color=INK, anchor="middle", weight=400):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{content}</text>'
    )


def node(point, label, color=INK, radius=14):
    x, y = point
    return (
        f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="#ffffff" stroke-width="2"/>'
        + text(x, y + 5, label, 13, "#ffffff", weight=700)
    )


def hand_map():
    connections = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
    palm = " ".join(f"{HAND_POINTS[i][0]},{HAND_POINTS[i][1]}" for i in (0, 1, 5, 9, 13, 17))
    body = [f'<polygon points="{palm}" fill="{SOFT}" stroke="{SOFT}" stroke-width="44" stroke-linejoin="round"/>']
    body += [line(HAND_POINTS[a], HAND_POINTS[b], SOFT, 44) for a, b in connections]
    colors = {0: INK}
    for start, name in zip((1, 5, 9, 13, 17), FINGERS):
        colors.update({i: FINGERS[name] for i in range(start, start + 4)})
    body += [line(HAND_POINTS[a], HAND_POINTS[b], colors[max(a, b)], 3) for a, b in connections]
    body += [node(HAND_POINTS[i], str(i), colors[i]) for i in sorted(HAND_POINTS)]
    x = 74
    for name, color in {"Wrist": INK, **FINGERS}.items():
        body.append(f'<circle cx="{x}" cy="656" r="7" fill="{color}"/>')
        body.append(text(x + 13, 661, name, 14, anchor="start"))
        x += 92
    return svg(620, 680, "Hand landmark map with ids 0 to 20", body)


def pose_map():
    connections = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
    body = [
        '<rect x="572" y="96" width="316" height="310" rx="12" fill="#f4f8f7"/>',
        text(730, 126, "Face points, enlarged", 15, MUTED),
        f'<ellipse cx="730" cy="262" rx="124" ry="118" fill="{SOFT}"/>',
        f'<circle cx="280" cy="92" r="52" fill="{SOFT}"/>',
        text(280, 98, "0 to 10", 15, MUTED),
    ]
    for a, b in connections:
        if a >= 11 and b >= 11:
            color = INK if (a, b) in ((11, 12), (23, 24)) else LEFT_SIDE if a % 2 else RIGHT_SIDE
            body.append(line(POSE_BODY[a], POSE_BODY[b], color, 3))
        elif a <= 10 and b <= 10:
            body.append(line(POSE_FACE[a], POSE_FACE[b], MUTED, 2))
    for i, point in POSE_BODY.items():
        body.append(node(point, str(i), LEFT_SIDE if i % 2 else RIGHT_SIDE))
    for i, point in POSE_FACE.items():
        body.append(node(point, str(i), INK, 13))
    body.append(text(100, 40, "Person's right", 16, RIGHT_SIDE, weight=600))
    body.append(text(460, 40, "Person's left", 16, LEFT_SIDE, weight=600))
    return svg(900, 890, "Pose landmark map with ids 0 to 32", body)


def face_map():
    body = [f'<ellipse cx="240" cy="270" rx="172" ry="212" fill="{SOFT}"/>']
    for i, (name, x, y) in FACE_POINTS.items():
        body.append(node((x, y), str(i), INK, 15))
        body.append(text(x, y + 34, name, 15))
    return svg(480, 510, "Face key point map with ids 0 to 5", body)


def face_mesh_map(photo):
    frame = cv.imread(photo)
    with FaceMesh(static_image=True, max_faces=1) as mesh:
        faces = mesh.detect(frame, draw=False)
    points = faces[0].points
    left, right = min(p[0] for p in points), max(p[0] for p in points)
    top, bottom = min(p[1] for p in points), max(p[1] for p in points)
    scale = 560 / (bottom - top)
    margin_x, margin_y = 250, 30

    def place(i):
        return (round((points[i][0] - left) * scale + margin_x, 1), round((points[i][1] - top) * scale + margin_y, 1))

    def segments(connections):
        # The tessellation lists most edges twice, once for each triangle that shares them.
        edges = sorted({tuple(sorted((c.start, c.end))) for c in connections})
        placed = ((place(a), place(b)) for a, b in edges)
        return "".join(f"M{a[0]} {a[1]}L{b[0]} {b[1]}" for a, b in placed)

    width = round((right - left) * scale + 2 * margin_x)
    tessellation = segments(FACE.FACE_LANDMARKS_TESSELATION)
    body = [f'<path d="{tessellation}" stroke="#c5d4d1" stroke-width="0.7" fill="none"/>']
    for connections, color in MESH_REGIONS.values():
        path = segments(connections)
        body.append(f'<path d="{path}" stroke="{color}" stroke-width="2" fill="none" stroke-linecap="round"/>')

    center = width / 2
    top_down = sorted(MESH_LABELS, key=lambda i: place(i)[1])
    sides = (
        (True, [i for i in top_down if place(i)[0] < center]),
        (False, [i for i in top_down if place(i)[0] >= center]),
    )
    for is_left, ids in sides:
        x = 196 if is_left else width - 196
        for row, i in enumerate(ids):
            y = 44 + 540 * (row + 0.5) / len(ids)
            point = place(i)
            body.append(line(point, (x, y), MUTED, 1))
            body.append(f'<circle cx="{point[0]}" cy="{point[1]}" r="4" fill="{INK}"/>')
            anchor = "end" if is_left else "start"
            gap = -8 if is_left else 8
            body.append(text(x + gap, y - 2, MESH_LABELS[i], 14, anchor=anchor))
            body.append(text(x + gap, y + 15, f"id {i}", 13, MUTED, anchor=anchor, weight=600))
    return svg(width, 620, "Face mesh map with key landmark ids", body)


def describe_hand(name):
    if name == "WRIST":
        return "Wrist"
    finger, _, joint = name.rpartition("_")
    finger = finger.replace("_FINGER", "").capitalize()
    joints = {
        "CMC": "base", "MCP": "knuckle", "IP": "middle joint", "PIP": "lower middle joint",
        "DIP": "upper middle joint", "TIP": "tip",
    }
    return f"{finger} {joints[joint]}"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def region_ids(connections):
    return ", ".join(map(str, sorted({i for c in connections for i in (c.start, c.end)})))


def main():
    MAPS.mkdir(parents=True, exist_ok=True)
    SNIPPETS.mkdir(parents=True, exist_ok=True)
    landmark_headers = ("Id", "Name", "Description")
    outputs = {
        MAPS / "hand.svg": hand_map(),
        MAPS / "pose.svg": pose_map(),
        MAPS / "face.svg": face_map(),
        MAPS / "face-mesh.svg": face_mesh_map(sys.argv[1]),
        SNIPPETS / "hand-landmarks.md": table(
            landmark_headers, ((m.value, f"`{m.name}`", describe_hand(m.name)) for m in HandLandmark)
        ),
        SNIPPETS / "pose-landmarks.md": table(
            landmark_headers, ((m.value, f"`{m.name}`", m.name.replace("_", " ").capitalize()) for m in PoseLandmark)
        ),
        SNIPPETS / "face-keypoints.md": table(
            ("Id", "Key point"), ((i, name) for i, (name, _, _) in FACE_POINTS.items())
        ),
        SNIPPETS / "face-mesh-regions.md": table(
            ("Region", "Landmark ids"), ((name, region_ids(conns)) for name, (conns, _) in MESH_REGIONS.items())
        ),
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf8")


if __name__ == "__main__":
    main()
