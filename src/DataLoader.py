import os
import cv2
import json
import random
import numpy as np

from src.SamplePreprocessor import preprocess


class Sample:

    def __init__(self, gt_text, file_path):
        self.gtText = gt_text
        self.filePath = file_path


class Batch:

    def __init__(self, gt_texts, images):
        self.imgs = np.stack(images, axis=0)
        self.gtTexts = gt_texts


class DataLoader:

    def __init__(self, batchSize, imgSize, maxTextLen, train_dir):
        self.dataAugmentation = False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []

        self.split = .95

        chars = set()
        for location in os.listdir(train_dir):
            ground_truth_location = os.path.join(train_dir, location, 'ground_truth.json')

            if not os.path.exists(ground_truth_location):
                continue

            with open(ground_truth_location) as json_file:
                ground_truth = json.loads(json_file.read())

            images_dir = os.path.join(train_dir, location, 'images')
            for image_location in os.listdir(images_dir):
                full_image_location = os.path.join(images_dir, image_location)

                if os.path.exists(full_image_location):
                    image_id = str(image_location.split('.')[0])

                    gt = ground_truth.get(image_id)
                    if gt is None:
                        continue

                    assert all(map(lambda x: x in '.0123456789', gt))

                    gt = self.truncateLabel(gt, maxTextLen)
                    chars |= set(list(gt))

                    # put sample into list
                    self.samples.append(Sample(gt, full_image_location))

        random.shuffle(self.samples)

        # split into training and validation set
        split_index = int(self.split * len(self.samples))
        self.trainSamples = self.samples[:split_index]
        self.validationSamples = self.samples[split_index:]

        # put words into lists
        self.trainWords = [x.gtText for x in self.trainSamples]
        self.validationWords = [x.gtText for x in self.validationSamples]

        # start with train set
        self.trainSet()

        # list of all chars in dataset
        self.charList = sorted(list(chars))

    def truncateLabel(self, text, maxTextLen):
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text

    def trainSet(self):
        self.currIdx = 0
        self.dataAugmentation = True
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples

    def validationSet(self):
        self.currIdx = 0
        self.dataAugmentation = False
        self.samples = self.validationSamples

    def getIteratorInfo(self):
        return self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize

    def hasNext(self):
        return self.currIdx + self.batchSize <= len(self.samples)

    def getNext(self):
        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        gtTexts = [self.samples[i].gtText for i in batchRange]
        imgs = [
            preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation)
            for i in batchRange]
        self.currIdx += self.batchSize

        return Batch(gtTexts, imgs)
