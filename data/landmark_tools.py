import cv2

from data.face_rect_dectator import get_facebox


def read_points(file_name=None):
    """
    Read points from .pts file.
    """
    points = []
    with open(file_name) as file:
        line_count = 0
        for line in file:
            if "version" in line or "points" in line or "{" in line or "}" in line:
                continue
            else:
                loc_x, loc_y = line.strip().split(sep=" ")
                points.append([float(loc_x), float(loc_y)])
                line_count += 1
    return points


def draw_landmark_point(image, points):
    """
    Draw and show landmark point of the image.
    """
    for point in points:
        cv2.circle(image, (int(point[0]), int(
            point[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.imshow("result", image)
    cv2.waitKey(0)


def show_points_and_sqr(image, box, points):
    """ show boxs and points on an image"""
    label = "face"
    cv2.rectangle(image, (box[0], box[1]),
                  (box[2], box[3]), (0, 255, 0))
    label_size, base_line = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    cv2.rectangle(image, (box[0], box[1] - label_size[1]),
                  (box[0] + label_size[0],
                   box[1] + base_line),
                  (0, 255, 0), cv2.FILLED)
    cv2.putText(image, label, (box[0], box[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    for point in points:
        cv2.circle(image, (int(point[0]), int(
            point[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.imshow("result", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    points_read = read_points("D:\\training_data\\video\\300VW_Dataset_2015_12_14\\002\\images\\002_000100.pts")
    image_read = cv2.imread("D:\\training_data\\video\\300VW_Dataset_2015_12_14\\002\\images\\002_000100.jpg")
    box = get_facebox(image=image_read)[-1][0]
    show_points_and_sqr(image_read, box, points_read)
