from numpy.lib.type_check import imag
from ppadb.client import Client
import cv2
import pytesseract
from BotDevice import BotDevice
import os
import numpy as np
import re
import SudokuSolver
import time
import itertools
import copy


from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} took: {te-ts} sec')
        return result
    return wrap

os.environ["TESSDATA_PREFIX"] = r"E:\Desktop\Programming\Python\SudokuBot\tessdata"

def cropSudokuGrid(image, rect):
    """Returns an image bound by a rectangle."""

    return image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+(rect[2])]

def cropGridSquare(image, pos):
    return image[pos[1]:pos[1]+(pos[3]-pos[1]), pos[0]:pos[0]+(pos[2]-pos[0])]

def processImage(image):
    """Applies post-processing to an image to prepare it for OCR processing."""

    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    x, thresholded = cv2.threshold(grayscale, 100, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)
    dilated = cv2.erode(thresholded, kernel, 1)

    blur = cv2.medianBlur(dilated, 3)
    return blur

def getNumberLocationsFromImage(image, cols = 9, rows = 9, inset = 10):
    """Returns the locations of all number cells on a given image."""

    h,w = image.shape

    print(h,w)

    rects = []

    for i in range(rows):
        for j in range(cols):
            x1 = int(w/cols*j) + inset
            y1 = int(h/rows*i) + inset
            x2 = x1 + int(w/cols) - (inset * 2)
            y2 = y1 + int(h/rows) - (inset * 2)
            rects.append((x1, y1, x2, y2))

    return rects

def getText(image):
    """Extracts a single digit from the image."""

    orcResult = pytesseract.image_to_string(image, "digits", config='--psm 9 --oem 3')
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', orcResult)

def extractGridData(image, gridSquares):
    results = []

    for idx, square in enumerate(gridSquares):
        croppedGridSquare = cropGridSquare(image, square)
        text = getText(croppedGridSquare).replace('\n', '')

        if text == "":
            text = "0"
        results.append(int(text))

        cv2.rectangle(sudokuGrid, (square[0], square[1]), (square[2], square[3]), color=(0,0,255), thickness=2)
        cv2.imshow("Phone Screen", cv2.resize(sudokuGrid, (0,0), fx=0.4, fy=0.4))
        cv2.waitKey(1)

        print(f"Processed Cell: {idx+1}/81", end="\r")

    return list(chunks(results, 9))

def annotateSolvedGrid(image, gridSquares, mergedSolvedGrid, mergedUnsolvedGrid):
    for idx, pos in enumerate(gridSquares):
        if mergedUnsolvedGrid[idx] == 0:
            midpoint = ((pos[0] + pos[2])/2, (pos[1] + pos[3])/2)
            cv2.putText(image, str(mergedSolvedGrid[idx]), (int(midpoint[0]) - 5, int(midpoint[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)

    cv2.imshow("Phone Screen", cv2.resize(sudokuGrid, (0,0), fx=0.4, fy=0.4))
    cv2.waitKey(1)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

#Initialize a BotDevice
adb = Client(host='127.0.0.1', port=5037)
bot = BotDevice(adb, 0)

#Screenshot the phone and crop out the sudoku grud
image = bot.screenshot()
sudokuGrid = cropSudokuGrid(image, (25, 455, 1035, 1035))

#Apply some image processing and extract the loactions of the cells
processedImage = processImage(sudokuGrid)

#Display images
cv2.imshow("Processed Screen", cv2.resize(processedImage, (0,0), fx=0.25, fy=0.25))

locations = getNumberLocationsFromImage(processedImage)


#Perform OCR on the processed grid image and extract all numbers
unsolvedGrid = extractGridData(processedImage, locations)
originalUnsolvedGrid = copy.deepcopy(unsolvedGrid)

#Solve the puzzle
SudokuSolver.solveSudoku(unsolvedGrid)

#Merge the 2D arrays into 2 seperate arrays
mergedUnsolvedGrid = list(itertools.chain.from_iterable(originalUnsolvedGrid))
mergedSolvedGrid = list(itertools.chain.from_iterable(unsolvedGrid))
annotateSolvedGrid(sudokuGrid, locations, mergedSolvedGrid, mergedUnsolvedGrid)

#Keycodes for the ADB keyevent
keyCodes = [7,8,9,10,11,12,13,14,15,16]

#Iterate through each cell. If the cell was occupied before solving it'll be skipped, else tap the cell and input the correct value
for idx, pos in enumerate(locations):
    number = keyCodes[mergedSolvedGrid[idx]]
    midpoint = ((pos[0] + pos[2])/2, (pos[1] + pos[3])/2)

    if mergedUnsolvedGrid[idx] == 0:
        bot.device.shell(f"input tap {25 + midpoint[0]} {455 + midpoint[1]}")
        bot.device.shell(f"input keyevent {number}")

    cv2.rectangle(sudokuGrid, (pos[0], pos[1]), (pos[2], pos[3]), color=(0,200,0), thickness=2)
    cv2.imshow("Phone Screen", cv2.resize(sudokuGrid, (0,0), fx=0.4, fy=0.4))
    cv2.waitKey(1)