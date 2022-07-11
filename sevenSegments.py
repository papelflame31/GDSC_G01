import cv2
import numpy as np
# Lectura de los segmentos de la imagen

class segments:
    def __init__(self):
        # Define los rangos donde se buscara detectar los segmentos
        self.f_segmento = []
        x = 0.10
        y = 0.02
        hor_1 = [[0.00, 1.00], [0.00 + y, 0.10 + y]]
        hor_2 = [[0.00, 1.00], [0.45 - y, 0.55 + y]]
        hor_3 = [[0.00, 1.00], [0.90 - y, 1.00 - y]]
        viz_1 = [[0.00 + x, 0.20 + x], [0.00, 0.50]]
        viz_2 = [[0.00 + x, 0.20 + x], [0.50, 1.00]]
        vde_1 = [[0.80 - x, 1.00 - x], [0.00, 0.50]]
        vde_2 = [[0.80 - x, 1.00 - x], [0.50, 1.00]]
        self.segmentos = [hor_1, hor_2, hor_3, 
        viz_1, viz_2, vde_1, vde_2]

    def imagenADigito(self, img_gray):
        # Para una imagen BINARIZADA detecta los segmentos con un  
        # porcentaje alto de pixeles NEGROS
        self.f_segmento = []

        h, w = img_gray.shape[:2]
        if(w < 0.3 * h): # Para el 1 (caso especial)
            self.f_segmento.append(5)
            self.f_segmento.append(6)
            return

        img = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)

        for idx, segmento in enumerate(self.segmentos): 
            inix, finx = segmento[0]
            iniy, finy = segmento[1]
            inix = int(inix * w)
            finx = int(finx * w)
            iniy = int(iniy * h)
            finy = int(finy * h)
            sw = finx - inix
            sh = finy - iniy

            count = np.count_nonzero(img_gray[iniy:finy, inix:finx] > 50)
            if (count / (sw * sh) > 0.6):
                self.f_segmento.append(idx)

    def getNum(self):
        # Devuelve el numero o -1 segun los segmentos detectados
        if self.f_segmento == [0, 2, 3, 4, 5, 6]:
            return 0
        if self.f_segmento == [5, 6]:
            return 1
        if self.f_segmento == [0, 1, 2, 4, 5]:
            return 2
        if self.f_segmento == [0, 1, 2, 5, 6]:
            return 3
        if self.f_segmento == [1, 3, 5, 6]:
            return 4
        if self.f_segmento == [0, 1, 2, 3, 6]:
            return 5
        if self.f_segmento == [0, 1, 2, 3, 4, 6]:
            return 6
        if self.f_segmento == [0, 5, 6]:
            return 7
        if self.f_segmento == [0, 1, 2, 3, 4, 5, 6]:
            return 8
        if self.f_segmento == [0, 1, 2, 3, 5, 6]:
            return 9
        return -1

def number_reading (img):
    seg = segments()
    seg.imagenADigito(img)
    return seg.getNum()