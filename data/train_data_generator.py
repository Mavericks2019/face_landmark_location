import cv2
import json

from file_manager import DirManager
from data.face_rect_dectator import get_facebox
from data.landmark_tools import read_points


def json_generator(file_path):
    image = cv2.imread(file_path)
    box = get_facebox(image=image)[-1][0]
    pts_path = file_path.split(".")[0] + ".pts"
    json_path = file_path.split(".")[0] + ".json"
    points = read_points(pts_path)
    save_dict = {
        "image_path": file_path,
        "box": box,
        "points": points
    }
    with open(json_path, "w") as js:
        json.dump(save_dict, js)


if __name__ == '__main__':

    file_manager = DirManager("D:\\training_data\\dirty_data")
    file_list = file_manager.file_filter({"png", "jpg"})
    for file in file_list:
        try:
            json_generator(file)
        except Exception as e:
            print(file + "error")
