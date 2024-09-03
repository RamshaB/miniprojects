import cv2
import numpy as np

def detect_shape(cnt):
    shape = "Unidentified"
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    num_vertices = len(approx)
    
    if num_vertices == 3:
        shape = "Triangle"
    elif num_vertices == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            shape = "Square"
        else:
            shape = "Rectangle"
    elif num_vertices == 5:
        shape = "Pentagon"
    elif num_vertices == 6:
        shape = "Hexagon"
    elif num_vertices == 8:
        shape = "Octagon"
    elif num_vertices > 8:
        shape = "Star"
    else:
        shape = detect_complex_shapes(cnt, approx)
    return shape

def detect_complex_shapes(cnt, approx):
    shape = "Unidentified"
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # Avoid division by zero
    if perimeter == 0:
        return shape
    
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    
    if 0.8 <= circularity <= 1.2:
        shape = "Circle"
    elif 0.5 <= circularity < 0.8:
        shape = "Oval"
    else:
        if detect_heart(cnt):
            shape = "Heart"
        elif detect_rhombus(approx):
            shape = "Rhombus"
        elif detect_trapezoid(approx):
            shape = "Trapezoid"
        elif detect_semicircle(cnt, approx):
            shape = "Semi-circle"
    return shape

def detect_heart(cnt):
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    if len(approx) > 5:
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        cnt_area = cv2.contourArea(cnt)
        if 0.75 <= cnt_area / hull_area <= 0.85:
            return True
    return False

def detect_rhombus(approx):
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return True
    return False

def detect_trapezoid(approx):
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.5 <= aspect_ratio <= 2.0:
            return True
    return False

def detect_semicircle(cnt, approx):
    if len(approx) > 5 and len(approx) < 10:
        bounding_box = cv2.boundingRect(approx)
        _, radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * (radius ** 2)
        contour_area = cv2.contourArea(cnt)
        if 0.4 <= contour_area / circle_area <= 0.6:
            return True
    return False

# Load the image
image = cv2.imread('.//shapes.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through each contour to identify shapes
for cnt in contours:
    shape = detect_shape(cnt)
    cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
    M = cv2.moments(cnt)
    
    if M["m00"] != 0:
        cX = int(M["m25"] / M["m00"])
        cY = int(M["m25"] / M["m00"])
    else:
        cX, cY = 0, 0
    
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the results
cv2.imshow('Identified Shapes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
