import os
import cv2
import logging
import requests
import numpy as np

hostname, port = '127.0.0.1', '5000'
url = 'http://{hostname}:{port}/recognize'.format(hostname=hostname, port=port)


def test_recognize():
    image_dir = '../images'

    h, bw = 175, 25
    files, images = {}, []
    for i, image in enumerate(filter(lambda x: x != 'interpretations.png', os.listdir(image_dir))):
        img = cv2.imread(os.path.join(image_dir, image), cv2.IMREAD_GRAYSCALE)
        images.append(cv2.cvtColor(cv2.resize(img, dsize=(275, h)), cv2.COLOR_GRAY2RGB))

        file_name = '{i}.png'.format(i=i)
        _, img_encoded = cv2.imencode('.png', img)
        files[file_name] = (file_name, img_encoded.tostring(), 'image/png')

    response = requests.post(url, files=files).json()
    for image, result in zip(images, response['result']):
        ox, oy = image.shape[0] // 10 * 9, image.shape[1] // 10
        cv2.putText(image, result['detected'], (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out_images = []
    for i, image in enumerate(images):
        out_images.append(image)
        if 0 <= i < len(images) - 1:
            out_images.append(np.full((h, bw, 3), 255, dtype=np.uint8))

    out_image = np.hstack(tuple(out_images))
    out_image_bordered = np.full((out_image.shape[0] + 2 * bw, out_image.shape[1] + 2 * bw, 3), 255, dtype=np.uint8)
    out_image_bordered[bw:-bw, bw:-bw] = out_image

    cv2.imshow('interpretations', out_image_bordered)
    cv2.waitKey()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_recognize()
