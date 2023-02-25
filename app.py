import cv2 as cv
import numpy as np
from keras.models import load_model
import tensorflow

#------------------------------------------------------------------------------------
# DEFS
def predict(file):          
        
        file = cv.resize(file, (img_x, img_y), interpolation = cv.INTER_AREA)
        imagemVerificar = tensorflow.keras.utils.img_to_array(file)
        imagemVerificar = np.expand_dims(imagemVerificar, axis = 0)
        result = classifier.predict(imagemVerificar)
        #print(result)
        maior, class_index = -1, -1
        for x in range(classes):      
            
            if result[0][x] > maior:
                maior = result[0][x]
                class_index = x
        
        return [result, letters[str(class_index)], maior]

#------------------------------------------------------------------------------------
#Variables

img_x, img_y = 32,32
classifier = load_model('models/model_cow_20230215_1634.h5', compile=False) # MELHOR MODELO
classes = 3
letters = {'0' : 'Vaca branca', '1' : 'Vaca malhada', '2' : 'Vaca preta'}
total = success = 0
#--------------------------------------------------------------------------------- ---
# Main

cows = ['branca', 'preta', 'malhada']

def teste_treinamento():
    for c in cows:
        cow = cv.imread(f"temp/{c}.png")

        cv.imshow(f"Vaca {c}", cow)

        result = predict(cow)
        result_percent = str(round(float(result[2]) * 100, 2))
        result_class = result[1]
        print(f'Vaca {c} --> Result: {result_class} with {result_percent}% accuracy')
        cv.waitKey(0)


def teste_imagens_pasto():
    pasto = cv.imread("temp/pasto2.png")

    pasto_process = cv.cvtColor(pasto, cv.COLOR_BGR2GRAY)
    pasto_process = cv.GaussianBlur(pasto_process, (7,7), 1)
    _, pasto_process = cv.threshold(pasto_process ,70, 255, cv.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    #pasto_process = cv.dilate(pasto_process, kernel, iterations=2)

    #pasto_process = cv.Canny(pasto_process, 70, 255)

    contornos, _ = cv.findContours(pasto_process, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    ctrl = 0
    for c in contornos:
        perimeter = cv.arcLength (c, True)
        if perimeter > 100:
            (x, y, alt, lar) = cv.boundingRect(c)
            aprox = cv.approxPolyDP (c,perimeter + 0.2 , True )
            roi = pasto[y:y+lar, x:x+alt]
            cv.imwrite("temp/posprocess/" + str(ctrl) + ".png", roi)
            #ctrl += 1
            result = predict(roi)
            result_percent = int(round(float(result[2]) * 100, 0))
            result_class = result[1]
            if result_percent > 80:
                print(f' --> Result: {result_class} with {result_percent}% accuracy')
                cv.rectangle(pasto, (x, y), (x+alt, y+lar), (0,255,0), 1)
                cv.putText(
                            img = pasto,
                            text = f"{result_class} {result_percent}%",
                            org = (x , y+5),
                            fontFace = cv.FONT_HERSHEY_PLAIN,
                            fontScale = 0.8,
                            color = (255, 255, 0),
                            thickness = 1
                            )
            else:
                pass
                #cv.rectangle(pasto, (x, y), (x+alt, y+lar), (0,0,255), 1)


    cv.imshow("Pasto",pasto)
    cv.imshow("Pasto processado",pasto_process)
    cv.imwrite("Results/vacas-pasto.png", pasto)
    cv.waitKey(0)


def teste_video_pasto():
    
    cap = cv.VideoCapture("temp/pasto-video2.mp4")
    if (cap.isOpened()== False):  
        print("Error opening video  file")

    while(cap.isOpened()):
        ret ,pasto = cap.read() 
        if ret == True:

            pasto_process = cv.cvtColor(pasto, cv.COLOR_BGR2GRAY)
            #pasto_process = cv.GaussianBlur(pasto_process, (7,7), 1)


            kernel = np.array([[0, -1,  0],
                            [-1,  5, -1],
                                [0, -1,  0]])
            
            #pasto_process = cv.filter2D(src=pasto_process, ddepth=-1, kernel=kernel)
            _, pasto_process = cv.threshold(pasto_process ,150, 255, cv.THRESH_OTSU)
            #_, pasto_process = cv.threshold(pasto_process ,100, 150, cv.THRESH_TRUNC)
            #pasto_process = cv.adaptiveThreshold(pasto_process,180,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\cv.THRESH_BINARY,13,12)

            pasto_process = cv.Canny(pasto_process, 100, 255)

            contornos, _ = cv.findContours(pasto_process, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            for c in contornos:
                perimeter = cv.arcLength(c, False)
                if perimeter > 100:
                    (x, y, alt, lar) = cv.boundingRect(c)
                    cv.rectangle(pasto, (x, y), (x+alt, y+lar), (0,255,0), 2)
                

            #cv.drawContours(pasto, contornos, -1, (0,255,0), 2)


            cv.imshow("Pasto",pasto)
            cv.imshow("Pasto processado",pasto_process)
            #cv.imshow("img", img)
            if cv.waitKey(20) == 27: 
                break

        else:
            break


# --------------------------------------------------------------------------

teste_imagens_pasto()
