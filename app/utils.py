import cv2
from imutils.perspective import four_point_transform
import torch
import numpy as np
from PIL import Image, ImageOps
import json
import Levenshtein as lev
import os
from sklearn import preprocessing

height = 1800
width = 1000
crop_amount = 3

targets_file = os.path.join(os.path.join("data", "labels.txt"))
with open(targets_file, "r", encoding='utf_16') as f:
    targets_orig = [line.strip() for line in f.readlines()]
targets = [[c for c in x] for x in targets_orig]
targets_flat = [c for clist in targets for c in clist]

lbl_enc = preprocessing.LabelEncoder()
lbl_enc.fit(targets_flat)

def scanImage(image):

    image = cv2.resize(image, (width, height))
    orig_image = image.copy()

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert the image to gray scale
    blur = cv2.GaussianBlur(image, (5, 5), 0) # Add Gaussian blur
    edged = cv2.Canny(blur, 75, 200) # Apply the Canny algorithm to find the edges

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours) == 0:
        return orig_image

    # if the biggest contour area is less than 10% of the image area we assume that there is no document in the image
    if cv2.contourArea(contours[0]) < 0.1 * height * width:
        return orig_image

    # go through each contour
    for contour in contours:

        # we approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)

        # if we found a countour with 4 points we break the for loop
        # (we can assume that we have found our document)
        if len(approx) == 4:
            doc_cnts = approx
            break

    warped = four_point_transform(orig_image, doc_cnts.reshape(4, 2))

    h, w = warped.shape
    h_new = h - crop_amount
    w_new = w - crop_amount

    cropped = warped[crop_amount:h_new, crop_amount:w_new]

    return cropped


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
          
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    # print(preds)
    text_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        text_preds.append(remove_duplicates(tp))
    return text_preds

def find_word_bounding_rectangles(image, dilation_kernel_y, dilate_kernel_x):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to obtain a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate the image to capture small sub-characters
    kernel = np.ones((dilation_kernel_y * 5, dilate_kernel_x), 'uint8')
    dilate_img_lines = cv2.dilate(thresh, kernel, iterations=1)

    # Dilate the image to capture small sub-characters
    kernel = np.ones((dilation_kernel_y, dilate_kernel_x), 'uint8')
    dilate_img_words = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilate_img_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to remove small noise
    min_contour_area = 200  # Adjust this value according to your needs
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Sort bounding rectangles of line boxes in the desired order
    line_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    line_boxes = sorted(line_boxes, key=lambda rect: (rect[1], rect[0]))

    # Find bounding rectangles for each individual word within line boxes
    word_boxes = []
    for line_box in line_boxes:
        x, y, w, h = line_box
        line_region = dilate_img_words[y:y+h, x:x+w]

        # Find contours in the line region
        word_contours, _ = cv2.findContours(line_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area to remove small noise
        word_contours = [cnt for cnt in word_contours if cv2.contourArea(cnt) > min_contour_area]

        line_boxes = []
        # Find bounding rectangles for each word contour
        for word_contour in word_contours:
            # Adjust contour coordinates to the original image
            contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(word_contour)
            word_box = (x + contour_x, y + contour_y, contour_w, contour_h)
            line_boxes.append(word_box)
        
        line_boxes = sorted(line_boxes, key=lambda rect: rect[1])
        word_boxes.extend(line_boxes)

    return word_boxes

def resize_image(image, resize):
    image = Image.fromarray(image).convert("L")
    image = ImageOps.expand(image, border=(0, 0, 0, resize[0]-image.size[1]), fill=255)
    image = image.resize((resize[1], resize[0]), resample=Image.BILINEAR)
    # Invert the image
    image = ImageOps.invert(image)
    # image.show()
    image = np.array(image)
    image = np.rot90(image) 

    image = np.expand_dims(image, axis=0).astype(np.float32)

    return torch.tensor(image).clone().detach().float()

def load_all_words():
    with open("data/all_words.json", "r", encoding="utf-16") as f:
        all_words = json.load(f)
    return all_words

def correct_text(text, max_distance=3):
    """Returns the corrected text based on edit distance threshold"""
    words = text.split()
    corrected_words = []

    all_words = load_all_words()

    for word in words:
        if word in all_words:
            corrected_words.append(word)
        else:
            candidates = []
            for candidate in all_words:
                if abs(len(word) - len(candidate)) > max_distance:
                    continue
                distance = lev.distance(word, candidate)
                if distance <= max_distance:
                    candidates.append((candidate, distance))
            if candidates:
                corrected_word = min(candidates, key=lambda x: x[1])[0]
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
    corrected_text = " ".join(corrected_words)
    return corrected_text

def levenshtein_distance(word1, word2, substitute_costs):
    m, n = len(word1), len(word2)
    # Initialize a matrix of size (m+1) x (n+1) to store the distances
    distance = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column of the matrix
    for i in range(m + 1):
        distance[i][0] = i
    for j in range(n + 1):
        distance[0][j] = j

    # Compute the Levenshtein distance
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                cost = 0  # No substitution required
            else:
                cost = substitute_costs.get((ord(word1[i - 1]), ord(word2[j - 1])), 1)

            distance[i][j] = min(
                distance[i - 1][j] + 1,  # Deletion
                distance[i][j - 1] + 1,  # Insertion
                distance[i - 1][j - 1] + cost,  # Substitution
            )

    return distance[m][n]

def correct_text_w(text, max_distance=4):
    words = text.split()
    corrected_words = []

    all_words = load_all_words()

    substitute_costs = {}  # Substitute cost (dictionary of character pairs)

    substitute_costs[ord('ᠤ'), ord('ᠣ')] = 0.4
    substitute_costs[ord('ᠡ'), ord('ᠠ')] = 0.4
    substitute_costs[ord('ᠳ'), ord('ᠲ')] = 0.4
    substitute_costs[ord('ᠦ'), ord('ᠥ')] = 0.4
    substitute_costs[ord('ᠣ'), ord('ᠤ')] = 0.4
    substitute_costs[ord('ᠲ'), ord('ᠳ')] = 0.4
    substitute_costs[ord('ᠠ'), ord('ᠡ')] = 0.4
    substitute_costs[ord('ᠬ'), ord('ᠭ')] = 0.4
    substitute_costs[ord('ᠭ'), ord('ᠬ')] = 0.4
    substitute_costs[ord('ᠤ'), ord('ᠦ')] = 0.5
    substitute_costs[ord('ᠦ'), ord('ᠤ')] = 0.5
    substitute_costs[ord('ᠠ'), ord('ᠢ')] = 0.5
    substitute_costs[ord('ᠰ'), ord('ᠬ')] = 0.5
    substitute_costs[ord('ᠠ'), ord('ᠤ')] = 0.5
    substitute_costs[ord('ᠳ'), ord('ᠨ')] = 0.5
    substitute_costs[ord('ᠢ'), ord('ᠠ')] = 0.5
    substitute_costs[ord('ᠡ'), ord('ᠦ')] = 0.5
    substitute_costs[ord('ᠭ'), ord('ᠨ')] = 0.5
    substitute_costs[ord('ᠤ'), ord('ᠠ')] = 0.5
    substitute_costs[ord('ᠡ'), ord('ᠢ')] = 0.5
    substitute_costs[ord('ᠵ'), ord('ᠬ')] = 0.5
    substitute_costs[ord('ᠭ'), ord('ᠩ')] = 0.5
    substitute_costs[ord('ᠢ'), ord('ᠡ')] = 0.5
    substitute_costs[ord('ᠳ'), ord('ᠬ')] = 0.5
    substitute_costs[ord('ᠲ'), ord('ᠭ')] = 0.5
    substitute_costs[ord('ᠨ'), ord('ᠭ')] = 0.5
    substitute_costs[ord('ᠤ'), ord('ᠢ')] = 0.5
    substitute_costs[ord('ᠳ'), ord('ᠭ')] = 0.5
    substitute_costs[ord('ᠭ'), ord('ᠮ')] = 0.5
    substitute_costs[ord('ᠰ'), ord('ᠭ')] = 0.5
    substitute_costs[ord('ᠷ'), ord('ᠨ')] = 0.5
    substitute_costs[ord('ᠵ'), ord('ᠨ')] = 0.5
    substitute_costs[ord('ᠭ'), ord('ᠰ')] = 0.5
    substitute_costs[ord('ᠮ'), ord('ᠯ')] = 0.5
    substitute_costs[ord('ᠬ'), ord('ᠨ')] = 0.5
    substitute_costs[ord('ᠭ'), ord('ᠷ')] = 0.5
    substitute_costs[ord('ᠭ'), ord('ᠯ')] = 0.5
    substitute_costs[ord('ᠡ'), ord('ᠤ')] = 0.5
    substitute_costs[ord('ᠨ'), ord('ᠷ')] = 0.5
    substitute_costs[ord('ᠯ'), ord('ᠷ')] = 0.5
    substitute_costs[ord('ᠬ'), ord('ᠵ')] = 0.5
    substitute_costs[ord('ᠬ'), ord('ᠳ')] = 0.5
    substitute_costs[ord('ᠦ'), ord('ᠢ')] = 0.5
    substitute_costs[ord('ᠭ'), ord('ᠵ')] = 0.5
    substitute_costs[ord('ᠮ'), ord('ᠬ')] = 0.5
    substitute_costs[ord('ᠬ'), ord('ᠪ')] = 0.5
    substitute_costs[ord('ᠢ'), ord('ᠤ')] = 0.5
    substitute_costs[ord('ᠨ'), ord('ᠬ')] = 0.5
    substitute_costs[ord('ᠨ'), ord('ᠠ')] = 0.5
    substitute_costs[ord('ᠬ'), ord('ᠰ')] = 0.5

    for word in words:
        if word in all_words:
            corrected_words.append(word)
        else:
            candidates = []
            for candidate in all_words:
                if abs(len(word) - len(candidate)) > max_distance:
                    continue
                distance = levenshtein_distance(word, candidate, substitute_costs)
                if distance <= max_distance:
                    candidates.append((candidate, distance))
            if candidates:
                corrected_word = min(candidates, key=lambda x: x[1])[0]
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
    corrected_text = " ".join(corrected_words)
    return corrected_text