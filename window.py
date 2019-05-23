from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from lane_marking import LaneMarking

"""
Applies all the preprocessing to the input image
and returns the final result that includes the
distance to the center.
"""

def final_result():
    global panelL, panelR
    global btn_load, btn_step, btn_final
    global image_cv2
    global dim

    # Applies all the preprocessing
    clase = LaneMarking(image_cv2)
    clase.preprocess_image()
    clase.compute_lanes()
    clase.dissect_image()

    # Redimension the image
    image = clase.get_dissected_image()
    (h, w, d) = image.shape
    r = 400.0 / w
    dim = (400, int (h * r))
    resizedI = cv2.resize(image, dim)
    image = Image.fromarray(resizedI)
    image = ImageTk.PhotoImage(image)

    # If there was no right panel in the app, it creates it
    if panelR is None:
        panelR = Label(image=image)
        panelR.image = image
        panelR.pack(side = "right", padx = 10, pady = 10)
    # If there was right panel in the app, it replaces it
    else:
        panelR.configure(image=image)
        panelR.image = image
        panelR.pack(side = "right", padx = 10, pady = 10)

"""
Applies the next preprocessing step to the input image
and returns the corresponding result.
"""

def next_step():
    global panelL, panelR
    global btn_load, btn_step, btn_final, btn_next
    global image_cv2
    global c

    # Counter that indicates the preprocessing step to show
    c += 1

    # Applies all the preprocessing
    clase = LaneMarking(image_cv2)
    clase.preprocess_image()
    clase.compute_lanes()
    clase.dissect_image()

    if c == 1:
        # Show binarized image
        image = clase.get_binarized_image()

    elif c == 2:
        # Show edged image
        image = clase.get_edged_image()
    
    elif c == 3:
        # Show hough image
        image = clase.get_hough_image()
    
    elif c == 4:
        # Show image with detected lanes
        image = clase.get_lined_image()
    
    elif c == 5:
        # Show center and distances
        image = clase.get_dissected_image()

    # If a new image has been created
    if c > 0 and c < 6:
        # Resize the image
        resizedI = cv2.resize(image, dim)
        image = Image.fromarray(resizedI)
        image = ImageTk.PhotoImage(image)

        if panelR is None:
            # If there was no right panel in the app, it creates it
            panelR = Label(image=image)
            panelR.image = image
            panelR.pack(side = "right", padx = 10, pady = 10)
        else:
            # If there was right panel in the app, it replaces it
            panelR.configure(image=image)
            panelR.image = image
            panelR.pack(side = "right", padx = 10, pady = 10)
    
    # If it is the last preprocessing step
    if c >= 5:
        # Remove the next step button
        btn_next.pack_forget()

"""
Starts the step by step process with the first step,
the grayscale image.
"""

def start_steps():
    global panelL, panelR
    global btn_load, btn_step, btn_final, btn_next
    global image_cv2
    global c

    # Restarts the preprocessing step counter
    c = 0

    # Applies all the preprocessing
    clase = LaneMarking(image_cv2)
    clase.preprocess_image()
    clase.compute_lanes()
    clase.dissect_image()

    # Select the grayscale image and resizes it
    image = clase.get_grayscale_image()
    resizedI = cv2.resize(image, dim)
    image = Image.fromarray(resizedI)
    image = ImageTk.PhotoImage(image)

    # Add a next step button
    btn_next = Button(root, text="Next step", command=next_step)
    btn_next.pack(side = "right", fill = "both", expand = "yes", padx = "10", pady = "10")

    # If there was no right panel in the app, it creates it
    if panelR is None:
        panelR = Label(image=image)
        panelR.image = image
        panelR.pack(side = "right", padx = 10, pady = 10)

    # If there was right panel in the app, it replaces it
    else:
        panelR.configure(image=image)
        panelR.image = image
        panelR.pack(side = "right", padx = 10, pady = 10)
    
    # Removes the step by step and final result buttons
    btn_step.pack_forget()
    btn_final.pack_forget()

"""
Opens the file dialog to select and display the image to
be processed.
"""

def select_image():
    global panelL, panelR
    global btn_load, btn_step, btn_final, btn_next
    global image_cv2
    global dim

    # Shows the file selection dialog
    path = filedialog.askopenfilename()

    # If an image was selected
    if len(path) > 0:

        # Creates the image
        image_cv2 = cv2.imread(path)
        image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        # Resizes the image
        (h, w, d) = image.shape
        r = 400.0 / w
        dim = (400, int(h * r))
        resizedI = cv2.resize(image, dim)

        image = Image.fromarray(resizedI)
        image = ImageTk.PhotoImage(image)

        # If there is not a left panel in the app, it means it's
        # the first time we are using it.
        if panelL is None:
            # Displays the selected image in the left panel
            panelL = Label(image = image)
            panelL.image = image
            panelL.pack(side = "left", padx = "10", pady = "10")

        else:
            # Replace the old study image with the new selected one
            panelL.configure(image = image)
            panelL.image = image

            # Removes the right panel results
            if panelR is not None:
                panelR.pack_forget()

            # Remove the next step button
            if btn_next is not None:
                btn_next.pack_forget()

        # Add the step by step button
        btn_step = Button(root, text = "Step by step", command = start_steps)
        btn_step.pack(side = "right", padx = "10", pady = "10")

        # Add the final result button
        btn_final = Button(root, text = "Final result", command = final_result)
        btn_final.pack(side = "right", padx = "10", pady = "10")   

######################################################################

if __name__ == '__main__':

    root = Tk()
    panelL = None
    panelR = None

    btn_load = Button(root, text = "Select an image", command = select_image)
    btn_load.pack(side = "bottom", fill = "both", expand = "yes", padx = "10", pady = "10")

    root.mainloop()