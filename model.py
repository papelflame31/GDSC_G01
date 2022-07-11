import cv2
import time
import numpy as np
import functools
import utils

import imageio 
make_gif = []

# Lectura del video y tasa de lectura de frames
video = cv2.VideoCapture("Video de prueba.mp4")
frame_rate = 7
prev = 0
texto = ""

# Relacion ancho/largo a buscar
R_WIDHT_HEIGTH = 2.4

# Numero de digitos en la pantalla
N_DIGITS = 4
DECIMAL_POINT_POS = 3


# Captura de los frames
while(video.isOpened()):
    time_elapsed = time.time() - prev
    ret, image = video.read()
    
    if time_elapsed > 1./frame_rate:
        prev = time.time()
        if ret == True:
            # Redimencion del frame
            image = utils.image_resize(image, height = 480) 
            placa = []
                      
            # Ubicacion y busqueda del contornos de la pantalla
            canny = cv2.Canny(image,150,200)
            Contours,_ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in Contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                epsilon = 0.09 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                aspect_ratio = float(w) / h
                if aspect_ratio > R_WIDHT_HEIGTH:
                    if area > 1000:
                        placa = image[y: y + h, x: x + w]
                        placa_color = placa.copy()
                        placa = utils.image_clean(placa)
                        cv2.imshow('PLACA', utils.image_resize(placa, height = 80, width = 240))
                        
                        numeros = utils.image_segmentation(placa, placa_color, N_DIGITS)
                        actualizar = True
            
                        for numero in numeros:
                            if numero == -1:
                                actualizar = False
                                
                        if actualizar and len(numeros) == N_DIGITS:
                            texto = ""
                            for idx, numero in enumerate(numeros):
                                texto = texto + str(numero)
                                if(idx + 1 == DECIMAL_POINT_POS):
                                    texto = texto + "."
                        
                        cv2.putText(image, texto, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        break

            cv2.imshow('Resultado',image)
            make_gif.append(image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
imageio.mimsave('movie.gif', make_gif)
video.release()
cv2.destroyAllWindows()