from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from lane_marking import LaneMarking

def final_result():
    global panelA, panelB
    global btn_load, btn_step, btn_final
    global image_cv2
    global dim

    clase = LaneMarking(image_cv2)
    clase.preprocess_image()
    clase.compute_lanes()
    clase.dissect_image()

    image = clase.get_dissected_image()
    (h, w, d) = image.shape
    r = 400.0 / w
    dim = (400, int (h * r))
    resizedI = cv2.resize(image, dim)
    image = Image.fromarray(resizedI)
    image = ImageTk.PhotoImage(image)

    if panelB is None:
        panelB = Label(image=image)
        panelB.image = image
        panelB.pack(side = "right", padx = 10, pady = 10)
    else:
        panelB.configure(image=image)
        panelB.image = image
        panelB.pack(side = "left", padx = 10, pady = 10)

def next_step():
    global panelA, panelB, btn_load, btn_step, btn_final, image_cv2, c, btn_next
    c += 1

    clase = LaneMarking(image_cv2)
    clase.preprocess_image()
    clase.compute_lanes()
    clase.dissect_image()

    if c == 1:
        image = clase.get_binarized_image()

    elif c == 2:
        image = clase.get_edged_image()
    
    elif c == 3:
        image = clase.get_hough_image()
    
    elif c == 4:
        image = clase.get_lined_image()
    
    elif c == 5:
        image = clase.get_dissected_image()

    if c > 0 and c < 6:
        resizedI = cv2.resize(image, dim)
        image = Image.fromarray(resizedI)
        image = ImageTk.PhotoImage(image)

        if panelB is None:
            panelB = Label(image=image)
            panelB.image = image
            panelB.pack(side = "right", padx = 10, pady = 10)
        else:
            panelB.configure(image=image)
            panelB.image = image
            panelB.pack(side = "left", padx = 10, pady = 10)
    
    if c >= 5:
        btn_next.pack_forget()

        btn_step = Button(root, text="Step by step", command=start_steps)
        btn_step.pack(side="bottom", padx="10", pady="10")

        btn_final = Button(root, text="Final result", command=final_result)
        btn_final.pack(side="bottom", padx="10", pady="10")

def start_steps():
    global panelA, panelB, btn_load, btn_step, btn_final, image_cv2, c, btn_next
    c = 0

    clase = LaneMarking(image_cv2)
    clase.preprocess_image()
    clase.compute_lanes()
    clase.dissect_image()

    image = clase.get_grayscale_image()
    resizedI = cv2.resize(image, dim)
    image = Image.fromarray(resizedI)
    image = ImageTk.PhotoImage(image)

    if panelB is None:
        btn_next = Button(root, text="Next step", command=next_step)
        btn_next.pack(side = "bottom", fill = "both", expand = "yes", padx = "10", pady = "10")

        panelB = Label(image=image)
        panelB.image = image
        panelB.pack(side = "right", padx = 10, pady = 10)

    else:
        btn_next = Button(root, text="Next step", command=next_step)
        btn_next.pack(side = "bottom", fill = "both", expand = "yes", padx = "10", pady = "10")

        panelB.configure(image=image)
        panelB.image = image
        panelB.pack(side = "right", padx = 10, pady = 10)
    
    btn_step.pack_forget()
    btn_final.pack_forget()

def select_image():
    global panelA, panelB
    global btn_load, btn_step, btn_final, btn_next
    global image_cv2
    global dim

    path = filedialog.askopenfilename()

    if len(path) > 0:

        image_cv2 = cv2.imread(path)
        image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        (h, w, d) = image.shape
        r = 400.0 / w
        dim = (400, int(h * r))
        resizedI = cv2.resize(image, dim)

        image = Image.fromarray(resizedI)
        image = ImageTk.PhotoImage(image)

        if panelA is None:
            btn_step = Button(root, text = "Step by step", command = start_steps)
            btn_step.pack(side = "bottom", padx = "10", pady = "10")

            btn_final = Button(root, text = "Final result", command = final_result)
            btn_final.pack(side = "bottom", padx = "10", pady = "10")

            panelA = Label(image = image)
            panelA.image = image
            panelA.pack(side = "left", padx = "10", pady = "10")            
        
        else:
            panelA.configure(image = image)
            panelA.image = image

            if panelB is not None:
                panelB.pack_forget()
                

######################################################################

root = Tk()
panelA = None
panelB = None

btn_load = Button(root, text = "Select an image", command = select_image)
btn_load.pack(side = "bottom", fill = "both", expand = "yes", padx = "10", pady = "10")

root.mainloop()