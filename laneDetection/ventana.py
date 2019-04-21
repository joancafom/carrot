import tkinter
from tkinter import filedialog
import PIL
from PIL import Image, ImageTk
import cv2
import numpy as np

def ventana_principal():
    main = tkinter.Tk()
    main.title('Reconocimiento de Carriles')
    main.geometry('350x200')
    
    def cargar_imagen():        
        path = filedialog.askopenfilename(filetypes=[("Image File",'.jpg')])
        pilImage = PIL.Image.open(path)
        
        [pilWidth, pilHeight] = pilImage.size
        ratio = 300.0 / pilWidth
        newHeight = int(pilHeight * ratio)
        resizedImage = pilImage.resize((300, newHeight), Image.ANTIALIAS)

        tkImage = ImageTk.PhotoImage(resizedImage)

        imageLabel=tkinter.Label(main,image = tkImage)
        imageLabel.image = tkImage
        imageLabel.pack()

        boton_carga.place_forget()
        
        def detectar_bordes():
            cvImage = cv2.imread(path)
            (h, w, d) = cvImage.shape

            roi = cvImage[h//2:h-20, 0:w]            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)            
            thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
            edged2 = cv2.Canny(thresh, 50, 150)   

            lines = cv2.HoughLinesP(edged2, 1, np.pi/100, 40, minLineLength=50, maxLineGap=10)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 3)

            r = 300.0 / w
            dim = (300, int(h * r))
            resized = cv2.resize(roi, dim)
            
            cv2.imshow("Hough", resized)
            cv2.waitKey(0)
        
        boton_deteccion = tkinter.Button(main, text ="Detectar bordes", command = detectar_bordes)
        boton_deteccion.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)
       
    boton_carga = tkinter.Button(main, text ="Cargar imagen", command = cargar_imagen)
    boton_carga.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)
    
    main.mainloop()  

if __name__ == '__main__':
    ventana_principal()