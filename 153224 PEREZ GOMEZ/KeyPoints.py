import cv2
import numpy as np
import math
import tkinter.filedialog
from tkinter import simpledialog
import tkinter as tk
from PIL import Image
from PIL import ImageTk
from matplotlib import pyplot as plt
from tkinter import messagebox
from matplotlib import pyplot as plt
import harris 
import transformations
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt

def get_image():
    global image_1,button1,button2,button3,panel
    file_name=tk.filedialog.askopenfilename()
    if len(file_name)>0:
        image_1=cv2.imread(file_name)
        image_2=cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
        image_2 = Image.fromarray(image_2)
        image_2 = ImageTk.PhotoImage(image_2)
        panel=tk.Label(image=image_2)
        panel.imag=image_2
        panel.pack(side="right", padx=10, pady=10)
        button1.config( state = 'active')
        button2.config( state = 'active')
        button3.config( state = 'active')
    else:
        messagebox.showinfo("informacion ","No seleciono ninguna imagen")     

def rotation():
    array_of_images=[]
    number=input_dialog("ingrese los grados a rotar")
    total_degrees=340
    degrees_to_rotate=number
    iteration=1
    summation_degrees=degrees_to_rotate
    array_of_images=[]
    array_of_M=[]
    while(degrees_to_rotate>0 and summation_degrees<=total_degrees):  
        M,image_rotated=transformations.rotate_image(image_1,summation_degrees)
        path='/Users/PC/Downloads/IA/2_corte/keyPoints_C2.Act_3/ImagenesRotadas/'
        cv2.imwrite(path+'imagen'+str(iteration)+'.jpg',image_rotated) 
        array_of_images.append(image_rotated)
        array_of_M.append(M)
        iteration=iteration+1
        summation_degrees=summation_degrees+degrees_to_rotate
    method_harris_rotation(array_of_images,"rotacion",array_of_M)

def input_dialog(text):
    try:
        root=tk.Tk()
        root.withdraw()
        value=simpledialog.askstring(title="grados",prompt=text)
        number=int(value)
        return number
    except:
        if value is None:
            print("algo malo a ocurrido")  
        else:
            messagebox.showinfo("informacion ","ingrese un valor entero") 
                      
def window_two():
    global entry_zero_root_two,entry_one_root_two,root_two
    root_two= tk.Tk()
    root_two.title("cordenadas")
    root_two.config(width=300,height=200)
    lbl =tk.Label(root_two, text="cordenada en X",font=("Arial Bold", 8))
    lbl_1=tk.Label(root_two,text="cordenada en Y",font=("Arial Bold", 8))
    entry_zero_root_two= tk.Entry(root_two)
    entry_one_root_two=tk.Entry(root_two)
    lbl.place(x=30,y=50)
    entry_zero_root_two.place(x=115, y=50)
    lbl_1.place(x=30,y=80)
    entry_one_root_two.place(x=115,y=80)
    btn=tk.Button(root_two, text="ingresar datos",command=input_data)
    btn.place(x=130,y=110)
    root_two.mainloop()

def input_data():
    input_x=entry_zero_root_two.get()
    input_y=entry_one_root_two.get()
    x=int(input_x)
    y=int(input_y)
    if  x>0 and y:
        root_two.destroy()
        array_of_images=[]
        array_of_images,coordinates_x,coordinates_y=transformations.shifting(image_1,x,y)
        method_harris_displaced(array_of_images,coordinates_x,coordinates_y,"desplazado")
    else:
        messagebox.showinfo("informacion ","ingrese valores positivos") 


def resized():
    array_of_resized=[]
    array_of_images=[]
    array_of_images=transformations.resized_image(image_1)
    array_of_resized.append(0.25)
    array_of_resized.append(0.5)
    array_of_resized.append(200)
    array_of_resized.append(400)
    method_harris_resize(array_of_images,array_of_resized,"escalado")
    

def method_harris_rotation(array_of_images,text,array):
    image_gray_1=harris.image_gray(image_1)
    dataset_1=harris.dataset(image_gray_1)
    pix_1,kp_1,coordinates_x_y_1=harris.get_points(image_1,dataset_1)
    percentages=[]
    total_percentage=100
    number_one=len(kp_1)
    image_gray_1=harris.draw_points(image_gray_1,np.array(coordinates_x_y_1))
    for i in range(len(array_of_images)):
        image_gray_2=harris.image_gray(array_of_images[i])
        dataset_2=harris.dataset(image_gray_2)
        pix_2,kp_2,coordinates_x_y_2=harris.get_points(array_of_images[i],dataset_2)
        print("pixeles rotados ",len(pix_2)," cordenadas rotados ",len(coordinates_x_y_2)," key points rotados ",len(kp_2))
        kp_1_transformed=harris.rotated_coords(coordinates_x_y_1,array[i])
        print("kp 1 tranformados ",len(kp_1_transformed))
        new_coords_kp1_transformed=harris.convert_to_coordinates(kp_1_transformed)
        coordinates_x_y_2=np.array(coordinates_x_y_2)
        list_x_y_1,list_x_y_2=harris.rotation_matches(coordinates_x_y_1,coordinates_x_y_2,new_coords_kp1_transformed)
        print("lista 1 ",len(list_x_y_1),"lista 2",len(list_x_y_2))
        image_gray_2=harris.draw_points(image_gray_2,np.array(coordinates_x_y_2))
        number_two=len(list_x_y_2)
        percentage=int(total_percentage*number_two/number_one) 
        percentages.append(percentage)
        paint_matches(list_x_y_1,list_x_y_2,image_gray_1,image_gray_2)
    graphics(percentages,"rotacion")   

def method_harris_displaced(array_of_images,coordinates_x,coordinates_y,text):
    image_gray_1=harris.image_gray(image_1)
    dataset_1=harris.dataset(image_gray_1)
    pix_1,kp_1,coordinates_x_y_1=harris.get_points(image_1,dataset_1)
    percentages=[]
    total_percentage=100
    number_one=len(kp_1)
    image_gray_1=harris.draw_points(image_gray_1,np.array(coordinates_x_y_1))
    for i in range(len(array_of_images)):
        image_gray_2=harris.image_gray(array_of_images[i])
        dataset_2=harris.dataset(image_gray_2)
        pix_2,kp_2,coordinates_x_y_2=harris.get_points(array_of_images[i],dataset_2)
        x=coordinates_x[i]
        y=coordinates_y[i]
        kp_transformed=harris.kp_transform_deplaced_image(coordinates_x_y_1,x,y)
        coordinates_x_y_1=np.array(coordinates_x_y_1)
        coordinates_x_y_2=np.array(coordinates_x_y_2)
        kp_transformed=np.array(kp_transformed)
        list_x_y_1,list_x_y_2=harris.matches(coordinates_x_y_1,coordinates_x_y_2,kp_transformed)
        image_gray_2=harris.draw_points(image_gray_2,np.array(coordinates_x_y_2))
        number_two=len(list_x_y_2)
        percentage=int(total_percentage*number_two/number_one) 
        percentages.append(percentage)
        paint_matches(list_x_y_1,list_x_y_2,image_gray_1,image_gray_2)
    graphics(percentages,"desplazamiento")     

def method_harris_resize( array_of_images,array_of_resized,text):
    image_gray_1=harris.image_gray(image_1)
    dataset_1=harris.dataset(image_gray_1)
    pix_1,kp_1,coordinates_x_y_1=harris.get_points(image_1,dataset_1)
    percentages=[]
    total_percentage=100
    number_one=len(kp_1)
    image_gray_1=harris.draw_points(image_gray_1,np.array(coordinates_x_y_1))
    for i  in range(len(array_of_images)):
        image_gray_2=harris.image_gray(array_of_images[i])
        dataset_2=harris.dataset(image_gray_2)
        pix_2,kp_2,coordinates_x_y_2=harris.get_points(array_of_images[i],dataset_2)
        scaled=array_of_resized[i]
        kp_transformed=harris.transformed_kp_scaled(image_1,array_of_images[i],np.array(coordinates_x_y_1),np.array(coordinates_x_y_2),scaled)  
        list_x_y_1,list_x_y_2=harris.matches(np.array(coordinates_x_y_1),np.array(coordinates_x_y_2),np.array(kp_transformed))
        image_gray_2=harris.draw_points(image_gray_2,np.array(coordinates_x_y_2))
        number_two=len(list_x_y_2)
        percentage=int(total_percentage*number_two/number_one) 
        percentages.append(percentage)
        paint_matches(list_x_y_1,list_x_y_2,image_gray_1,image_gray_2)
    graphics(percentages,"escalado")
     

def graphics(percentages,text):
    print(percentages)
    iteraciones=[]
    for i in range(len(percentages)):
        iteraciones.append(str(i))
    plt.bar(range(len(percentages)), percentages, edgecolor='black')
    plt.xticks(range(len(iteraciones)), iteraciones, rotation=60)
    plt.title(text+str(" imagenes "))
    plt.ylim(min(percentages)-1, max(percentages)+1)
    plt.ylabel("porcentajes de matchs")
    plt.show()

def paint_matches(list_x_y_1,list_x_y_2,image_gray_1,image_gray_2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
    list_x_y_1=np.array(list_x_y_1)
    list_x_y_2=np.array(list_x_y_2)
    for i in range(len(list_x_y_1)):
        xyA = (list_x_y_2[i,0],list_x_y_2[i,1])  # in axes coordinates derecha(x,y)
        xyB = (list_x_y_1[i,0],list_x_y_1[i,1])  # in axes coordinates izquierda(x,y)
        coordsA = "data"
        coordsB = "data"
        con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB,
                        axesA=ax2, axesB=ax1,
                        arrowstyle="->", shrinkB=10)
        ax2.add_artist(con)
    ax1.imshow(image_gray_1,cmap='gray')
    ax2.imshow(image_gray_2,cmap='gray')
    plt.show()   


root_two=None
entry_zero_root_two=None
entry_one_root_two=None
root= tk.Tk()
root.title("Harris")

button2= tk.Button(root, text="escalar la imagen ",command=resized ,state = 'disabled')
button2.pack(side="bottom", fill="both", expand="no", padx="10", pady="10")

button3= tk.Button(root, text="desplazar la imagen",command=window_two,state = 'disabled')
button3.pack(side="bottom", fill="both", expand="no", padx="10", pady="10")

button1= tk.Button(root, text="rotar la imagen ",command=rotation,state ='disabled')
button1.pack(side="bottom", fill="both", expand="no", padx="10", pady="10")

button=tk.Button(root, text="Seleccionar imagen", command=get_image)
button.pack(side="bottom", fill="both", expand="no", padx="10", pady="10")

root.mainloop()

