import cv2
import logging
import numpy as np
import tensorflow as tf

from src.Model import Model, DecoderType
from src.DataLoader import DataLoader, Batch
from src.SamplePreprocessor import preprocess


def get_edit_distance(s1, s2):
    s1_length, s2_length = len(s1), len(s2)

    if s1_length and s2_length:
        previous_row = list(range(s2_length + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        edit_distance = previous_row[-1]
    elif s1_length:
        edit_distance = s1_length
    elif s2_length:
        edit_distance = s2_length
    else:
        edit_distance = 0

    return edit_distance


def train(model, loader, early_stopping=15):
    epoch = 0
    no_improvement_since = 0
    best_char_error_rate = np.inf

    while True:
        epoch += 1
        logging.info('Training NN. Epoch: {epoch}.'.format(epoch=epoch))

        loader.trainSet()
        while loader.hasNext():
            batch = loader.getNext()
            loss = model.trainBatch(batch)

            batch_info_0, batch_info_1 = loader.getIteratorInfo()
            if not batch_info_0 % 100:
                logging.info('Batch: {batch}, Loss: {loss}.'.format(batch=batch_info_0, loss=loss))

        char_error_rate = validate(model, loader)
        if char_error_rate < best_char_error_rate:
            no_improvement_since = 0
            best_char_error_rate = char_error_rate

            model.save()
            logging.info('The character error rate has improved, model weights are saved.')
        else:
            logging.info('The character error rate has not improved, model weights are not saved.')
            no_improvement_since += 1

        if no_improvement_since >= early_stopping:
            logging.info('Error rate has not improved for {nr} epochs. Training is stopped.'.format(nr=early_stopping))
            break


def validate(model, loader):
    logging.info('Validating NN')
    loader.validationSet()

    num_word_ok, num_word_total = 0, 0
    num_char_err, num_char_total = 0, 0
    while loader.hasNext():
        batch_info_0, batch_info_1 = loader.getIteratorInfo()

        if not batch_info_0 % 100:
            logging.info('Batch: {batch} / {info}.'.format(batch=batch_info_0, info=batch_info_1))

        batch = loader.getNext()
        recognized, _ = model.inferBatch(batch)

        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gtTexts[i] == recognized[i] else 0
            num_word_total += 1

            dist = get_edit_distance(recognized[i], batch.gtTexts[i])

            num_char_err += dist
            num_char_total += len(batch.gtTexts[i])

    word_accuracy = num_word_ok / num_word_total
    char_error_rate = num_char_err / num_char_total

    log_err, log_acc = char_error_rate * 100, word_accuracy * 100
    logging.info('Character error rate: {err}. Word accuracy: {acc}.'.format(err=log_err, acc=log_acc))

    return char_error_rate


def infer(model, img_location):
    img = cv2.imread(img_location, cv2.IMREAD_GRAYSCALE)
    img = preprocess(img, Model.imgSize)

    batch = Batch(None, [img])
    recognized, probability = model.inferBatch(batch, True)

    logging.info('Recognized: {rec} with probability: {proba}.'.format(rec=recognized[0], proba=probability[0]))


def train_model(model, train_dir):
    loader = DataLoader(Model.batchSize, Model.imgSize, Model.maxTextLen, train_dir)
    logging.info('Model chars: {chars}.'.format(chars=loader.charList))
    train(model, loader)


def infer_model(char_list, infer_location):
    model = Model(char_list, DecoderType.BestPath)
    infer(model, infer_location)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s;%(levelname)s;%(message)s')

    vocabulary = '.0123456789'
    in_dir = 'C:/Users/FlorijnWim/PycharmProjects/htr-ctctcnn/traindata'
    output_model_dir = 'C:/Users/FlorijnWim/PycharmProjects/htr-ctctcnn/model_new'

    graph = tf.Graph()
    with graph.as_default():
        output_model = Model(vocabulary, graph, DecoderType.BestPath, model_dir=output_model_dir)
        train_model(output_model, in_dir)
