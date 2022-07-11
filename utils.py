import cv2
import numpy as np
import sevenSegments

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # redimenciona la imagen segun ancho, largo o ambos
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    
    # Redimencion segun la altura 
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    # Redimencion segun el ancho 
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # Resultado de imagen redimencionada
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def image_clean(img):
    # Eliminacion del ruido y binarizacion
    dst = cv2.fastNlMeansDenoisingColored(img, None, 15, 10, 7, 21) 
    gray_image = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
                      
    adaptiveThreshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    blur = cv2.medianBlur(adaptiveThreshold, 3)
    return blur

def image_segmentation(img, imgcolor, N_DIGITS = 4):

    _, labels = cv2.connectedComponents(img)
    mask = np.zeros(img.shape, dtype="uint8")

    total_pixels = img.shape[0] * img.shape[1]
    lower = total_pixels // 80
    upper = total_pixels // 15
    
    for (i, label) in enumerate(np.unique(labels)):
        if label == 0: # Fondo
            continue
        labelMask = np.zeros(img.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)
    
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    
    segmentos = []
    valores = []
    for rect in boundingBoxes:
        if (len(boundingBoxes) != N_DIGITS):
            return [-1 for i in range(N_DIGITS)]
        
        x, y, w, h = rect
        if ((w < 3) or (h < 5)) or (y > img.shape[0] // 2 or y < img.shape[0] // 6):
            continue

        cv2.rectangle(imgcolor, (x,y), (x+w,y+h), (0, 255, 0), 2)
        segmento = img[y: y + h, x: x + w]
        segmentos.append(segmento)
        valores.append(sevenSegments.number_reading(image_resize(segmento, 150)))
        
    return valores[::-1]