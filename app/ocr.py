#!/usr/bin/env python

import numpy as np
import cv2
import torch
import statistics
from cv2 import IMREAD_GRAYSCALE
import os

from utils import scanImage, decode_predictions, find_word_bounding_rectangles, resize_image, correct_text, lbl_enc
from models.printed import CRNN
from models.handwritten import HandwrittenMongolianModel


line_size = 32
vocab = list(range(0x1800, 0x180F)) + list(range(0x1810, 0x181A)) + list(range(0x1820, 0x1879)) + \
        list(range(0x1880, 0x18AB)) + [0x202F]
vocab = "B "+ "".join([chr(v) for v in vocab])  # B for Blank
idx2char = {idx: char for idx, char in enumerate(vocab)}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def image_resize(image, width=None, height=None, inter=cv2.INTER_LINEAR):
    """Resize image but keep the aspect ratio."""
    assert width is not None or height is not None

    (h, w) = image.shape[:2]
    if width is None:
        r = height / float(h)
        return cv2.resize(image, (int(w * r), height), interpolation=inter)

    r = width / float(w)
    return cv2.resize(image, (width, int(h * r)), interpolation=inter)


def load_model_from_checkpoint(checkpoint_file_name, use_gpu=False):
    """Load a pretrained CRNN model."""
    model = CRNN(line_size, 1, len(vocab), 256)
    checkpoint = torch.load(checkpoint_file_name, map_location='cpu' if not use_gpu else None)
    model.load_state_dict(checkpoint['state_dict'])
    model.float()
    model.eval()
    model = model.cuda() if use_gpu else model.cpu()
    return model


def line_segmentation(orig_img, dilate_kernel_x=3, dilate_kernel_y=30, aspect_ratio=0.25, median_width_threshold=0.7):
    """Line segmentation."""
    lines = []

    img = orig_img
    if img.mean() < 100:
        # black background? then invert
        img = (255 - img)

    # threshold and dilate
    _, threshed_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((dilate_kernel_y, dilate_kernel_x), np.uint8)
    dilated_img = cv2.dilate(threshed_img, kernel, iterations=1)

    # compute and sort contours
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # compute median contour width
    widths = []
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if aspect_ratio * h >= w:
            # Mongolian script is written vertically so height must be bigger than width
            widths.append(w)
    if (len(widths) == 0):
        return lines
    # filter out contours by width
    median_width = statistics.median(widths)
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if aspect_ratio * h >= w and (median_width * (1 - median_width_threshold) <= w <= median_width * (1 + median_width_threshold)):
            lines.append([(x, y), (x+w, y+h)])
    return lines


def ocr_printed(orig_img, lines, checkpoint_file_name, use_gpu=False):
    """OCR on segmented lines."""
    model = CRNN(line_size, 1, len(vocab), 256)
    checkpoint = torch.load(checkpoint_file_name, map_location='cpu' if not use_gpu else None)
    model.load_state_dict(checkpoint['state_dict'])
    model.float()
    model.eval()
    model = model.cuda() if use_gpu else model.cpu()
    torch.set_grad_enabled(False)

    result = []
    for line in lines:
        (x1, y1), (x2, y2) = line
        line_img = image_resize(np.array(np.rot90(orig_img[y1:y2, x1:x2])), height=line_size)

        inputs = torch.from_numpy(line_img / 255).float().unsqueeze(0).unsqueeze(0)
        outputs = model(inputs)
        prediction = outputs.softmax(2).max(2)[1]

        def to_text(tensor, max_length=None, remove_repetitions=False):
            sentence = ''
            sequence = tensor.cpu().detach().numpy()
            for i in range(len(sequence)):
                if max_length is not None and i >= max_length:
                    continue
                char = idx2char[sequence[i]]
                if char != 'B':  # ignore blank
                    if remove_repetitions and i != 0 and char == idx2char[sequence[i - 1]]:
                        pass
                    else:
                        sentence = sentence + char
            return sentence
        predicted_text = to_text(prediction[:, 0], remove_repetitions=True)
        result.append((line_img, predicted_text))

    return result

def ocr_handwritten(image_path, model_path):
    success = True
    # Load the OCR model
    model = HandwrittenMongolianModel(num_chars=37)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        model.to(DEVICE)
        EPOCH_START = checkpoint['epoch']

    # Load the image
    image = cv2.imread(image_path)

    # height, _, _ = image.shape
    # dilation = round(height * 0.08)
    # Binarize the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inverted_image = cv2.bitwise_not(binary_image)

    # Find word bounding rectangles in the image
    bounding_rectangles = find_word_bounding_rectangles(image, 11, 13)

    if len(bounding_rectangles) == 0:
        success = False

    # Initialize an empty list to store predicted words
    predicted_words = []

    # Iterate over the bounding rectangles
    for rect in bounding_rectangles:
        x, y, w, h = rect

        # Crop out the word image
        word_image = inverted_image[y:y+h, x:x+w]

        # Resize the word image to the desired size
        resized_image = resize_image(word_image, (300, 48))

        # Perform prediction using the OCR model
        with torch.no_grad():
            model.eval()
            inputs = resized_image.unsqueeze(0).to(DEVICE)
            outputs, _ = model(inputs) 

        # Decode the predicted text
        predicted_text = decode_predictions(outputs, lbl_enc)[0]

        # Append the predicted word to the list
        predicted_words.append(predicted_text)

    corrected_preds = [correct_text(text) for text in predicted_words]
    predicted_words = corrected_preds
    # Combine all predicted words into a single text
    combined_text = ' '.join(predicted_words)

    return combined_text, success

def recognize(image_path, printedOrHandwritten):
    success = True

    if printedOrHandwritten == 'Printed':
        image = cv2.imread(image_path, IMREAD_GRAYSCALE)
        image = scanImage(image)
        height, _ = image.shape
        dilation = round(height * 0.08)
        lines = line_segmentation(image, dilate_kernel_y=dilation)
        if len(lines) == 0:
            success = False
        res = []
        for _, recognized_text in ocr_printed(image, lines, 'data/printed_trained.pth', use_gpu=False):
            res.append(recognized_text)

        return '\n'.join(res), success
    else:
        return ocr_handwritten(image_path, 'data/handwritten_trained.pth')
    