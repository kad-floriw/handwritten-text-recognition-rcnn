import os
import cv2
import json
import ntpath
import zipfile
import logging
import numpy as np
import tensorflow as tf

from pathlib import Path
from src.DataLoader import Batch
from src.Model import Model, DecoderType
from src.SamplePreprocessor import preprocess

graph = tf.Graph()
with graph.as_default():
    model_dir = 'C:/Users/FlorijnWim/PycharmProjects/htr-ctctcnn/weights'
    model = Model('.0123456789', graph=graph, decoderType=DecoderType.BestPath, model_dir=model_dir)

IN_DIR, OUT_DIR = 'D:/verwerkt', 'C:/Users/FlorijnWim/PycharmProjects/htr-ctctcnn/traindata'


def get_rotation_matrix(rotation, img_w, img_h):
    (cX, cY) = (img_w // 2, img_h // 2)
    rotation_matrix = cv2.getRotationMatrix2D((cX, cY), rotation, 1)

    # Rotate the image by the residual offset
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    # compute the new bounding dimensions of the image
    n_width = int((img_h * sin) + (img_w * cos))
    n_height = int((img_h * cos) + (img_w * sin))

    # adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (n_width / 2) - cX
    rotation_matrix[1, 2] += (n_height / 2) - cY

    return rotation_matrix


def rotate_point(x, y, rotation_matrix):
    p = np.array([x, y, 1])
    p = np.dot(rotation_matrix, p)
    p_x, p_y = int(p[0]), int(p[1])

    return p_x, p_y


def rotate_image(img, rotation):
    # Rotate the image first.
    img_h, img_w = img.shape[:2]
    rotation_matrix = get_rotation_matrix(rotation, img_w, img_h)

    # Compute the new bounding dimensions of the image
    cos, sin = np.abs(rotation_matrix[0, 0]), np.abs(rotation_matrix[0, 1])
    n_width = int((img_h * sin) + (img_w * cos))
    n_height = int((img_h * cos) + (img_w * sin))

    img_cpy = cv2.warpAffine(img, rotation_matrix, (n_width, n_height))

    return img_cpy, rotation_matrix


def read_zip(zip_name):
    with zipfile.ZipFile(zip_name, 'r') as archive:
        ground_truth = {}

        base_dir = os.path.join(OUT_DIR, ntpath.basename(zip_name))
        prefix, postfix = 'observations/snapshots/latest/', '.latest.json'
        sketch_files = list(filter(lambda x: x.startswith(prefix) and x.endswith(postfix), archive.namelist()))

        for i, sketch_file in enumerate(sketch_files):
            sketch_name = sketch_file[len(prefix):-len(postfix)]

            names = archive.namelist()
            images_dir = os.path.join(base_dir, 'images')
            Path(images_dir).mkdir(parents=True, exist_ok=True)

            attachment_prefix, img_extension = 'observations/attachments/front/' + sketch_name, '.JPG'
            image_files = list(filter(lambda x: x.startswith(attachment_prefix) and x.endswith(img_extension), names))
            if len(image_files):
                image_file = image_files[0]

                file = archive.read(image_file)
                image = cv2.cvtColor(cv2.imdecode(np.frombuffer(file, np.uint8), 1), cv2.COLOR_BGR2GRAY)

                fh = archive.open(sketch_file, 'r')
                json_data = json.loads(fh.read())

                text = json_data.get('text', {})
                points = json_data.get('points', {})
                for text_id, text_box in text.items():

                    value = text_box.get('value')
                    point = text_box.get('point')
                    origin = text_box.get('origin')
                    bounding_box = text_box.get('box')
                    if not any(map(lambda x: x is None, (value, point, origin, bounding_box))):
                        value = value.strip()
                        [[text_x, text_y], [w, h], angle] = bounding_box

                        image_rotated, matrix = rotate_image(image, angle)
                        half_width, half_height = int(w // 2), int(h // 2)
                        text_x, text_y = rotate_point(text_x, text_y, matrix)
                        image_rotated = image_rotated[text_y - half_height: text_y + half_height,
                                                      text_x - half_width: text_x + half_width]

                        point_position, origin_position = points.get(point), points.get(origin)
                        if image_rotated.size and not any(map(lambda x: x is None, (point_position, origin_position))):
                            point_x, point_y = point_position.get('position')
                            origin_x, origin_y = origin_position.get('position')
                            point_x, point_y = rotate_point(point_x, point_y, matrix)
                            origin_x, origin_y = rotate_point(origin_x, origin_y, matrix)

                            delta_y, delta_x = origin_y - point_y, point_x - origin_x
                            origin_point_angle = np.degrees(np.arctan2(delta_y, delta_x))

                            if -45 < origin_point_angle < 45:
                                image_rotated, _ = rotate_image(image_rotated, 90)
                            elif 135 < origin_point_angle <= 180 or -180 <= origin_point_angle < -135:
                                image_rotated, _ = rotate_image(image_rotated, -90)
                            elif -135 < origin_point_angle < -45:
                                image_rotated, _ = rotate_image(image_rotated, 180)

                            img = preprocess(image_rotated, Model.imgSize)

                            batch = Batch(None, [img])
                            with graph.as_default():
                                recognized, _ = model.inferBatch(batch, True)

                            if value.replace('.', '') == recognized[0].replace('.', ''):
                                output_id = sketch_name + '_' + text_id
                                ground_truth[output_id] = value
                                cv2.imwrite(os.path.join(images_dir, output_id + '.tiff'), image_rotated)

        with open(os.path.join(base_dir, 'ground_truth.json'), 'w') as fp:
            json.dump(ground_truth, fp)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s;%(levelname)s;%(message)s')

    zip_files = os.listdir(IN_DIR)
    dir_size = len(zip_files)
    for j, zip_file_name in enumerate(zip_files):
        logging.info('Processing project: {index}/{size}: {name}.'.format(index=j+1, size=dir_size, name=zip_file_name))

        zip_location = os.path.join(IN_DIR, zip_file_name)
        read_zip(zip_location)
