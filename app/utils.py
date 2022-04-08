
import cv2
from imutils.perspective import four_point_transform

height = 1800
width = 1000
green = (0, 255, 0)
crop_amount = 3

def scanImage(image):

    # image = cv2.imread("5.png")
    image = cv2.resize(image, (width, height))
    orig_image = image.copy()

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert the image to gray scale
    blur = cv2.GaussianBlur(image, (5, 5), 0) # Add Gaussian blur
    edged = cv2.Canny(blur, 75, 200) # Apply the Canny algorithm to find the edges

    # Show the image and the edges
    # cv2.imshow('Original image:', image)
    # cv2.imshow('Edged:', edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # If you are using OpenCV v3, v4-pre, or v4-alpha
    # cv2.findContours returns a tuple with 3 element instead of 2
    # where the `contours` is the second one
    # In the version OpenCV v2.4, v4-beta, and v4-official
    # the function returns a tuple with 2 element 
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Show the image and all the contours
    # cv2.imshow("Image", image)
    # cv2.drawContours(image, contours, -1, green, 3)
    # cv2.imshow("All contours", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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

    # We draw the contours on the original image not the modified one
    # cv2.drawContours(orig_image, [doc_cnts], -1, green, 3)
    # cv2.imshow("Contours of the document", orig_image)
    # apply warp perspective to get the top-down view
    warped = four_point_transform(orig_image, doc_cnts.reshape(4, 2))
    # convert the warped image to grayscale
    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Scanned", cv2.resize(warped, (600, 800)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    h, w = warped.shape
    h_new = h - crop_amount
    w_new = w - crop_amount
    # print((h_new, w_new))
    cropped = warped[crop_amount:h_new, crop_amount:w_new]

    # cv2.imshow("Cropped", cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite('5_cropped.png', cropped)

    return cropped


# img = cv2.imread('6.png')
# cv2.imshow('Scanned', scanImage(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()