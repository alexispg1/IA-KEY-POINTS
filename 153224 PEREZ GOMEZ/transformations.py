import cv2
import numpy as np
from scipy import misc,ndimage


def rotate_image(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = ((w-1) // 2.0, (h-1)// 2.0)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY),angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    #print(nW, nH)
    # adjust the rotation matrix to take into account translation
    M[0, 2] += ((nW-1) / 2.0) - cX
    M[1, 2] += ((nH-1) / 2.0) - cY
    # perform the actual rotation and return the image
    return M, cv2.warpAffine(image, M, (nW, nH)) 


def shifting(image_1,x,y):
    coordinates_x=[]
    coordinates_y=[]
    height,width=image_1.shape[0:2]
    #(x) positive and (y) positive
    coordinates_x.append(x)
    coordinates_y.append(y)
    #(x) negative and (y) negative
    coordinate_x=x*-1
    coordinate_y=y*-1 
    coordinates_x.append(coordinate_x)
    coordinates_y.append(coordinate_y)
    # (x) positive and (y) negative
    coordinate_x=coordinate_x*-1
    coordinates_x.append(coordinate_x)
    coordinates_y.append(coordinate_y)
    # (x) negativo and (y) pisitivo
    coordinate_x=coordinate_x*-1
    coordinate_y=coordinate_y*-1
    coordinates_x.append(coordinate_x)
    coordinates_y.append(coordinate_y)
    #axis x positive
    axis_x_positive=x
    coordinates_x.append(axis_x_positive)
    coordinates_y.append(0)
    #axis x negative 
    axis_x_negative=axis_x_positive*-1
    coordinates_x.append(axis_x_negative)
    coordinates_y.append(0)
    #axis y positive
    axis_y_positive=y
    coordinates_x.append(0)
    coordinates_y.append(axis_y_positive)
    #axis y negative
    axis_y_negative=axis_x_positive*-1
    coordinates_x.append(0)
    coordinates_y.append(axis_y_negative)

    array_of_images=[]
    array_of_images=displacement_image_save(image_1,coordinates_x,coordinates_y,width,height)
    return array_of_images,coordinates_x,coordinates_y


def displacement_image_save(image_1,coordinates_x,coordinates_y,width,height):
    array_of_images=[]
    for i in range(len(coordinates_x)):
        matrix=np.float32([[1,0,coordinates_x[i]],[0,1,coordinates_y[i]]])
        shifting_image= cv2.warpAffine(image_1,matrix,(width,height))
        path='/Users/PC/Downloads/IA/2_corte/keyPoints_C2.Act_3/ImagenesDesplazadas/'
        cv2.imwrite(path+'image'+str(i)+'.jpg',shifting_image)
        array_of_images.append(shifting_image)
    return array_of_images    
   
def resized_image(image_1):
    array_of_images=[]
    number=4
    height,width =image_1.shape[:2] 
    for i in range(2):
            new_height,new_width=int(height/number),int(width/number)
            image_resized=escaled_up_image_save(image_1,new_width,new_height,i)
            array_of_images.append(image_resized)
            number=number-2
    number2=200
    number4=400
    new_height,new_width=int(number2*height/100),int(number2*width/100) 
    image_resized=escaled_up_image_save(image_1,new_width,new_height,2)
    array_of_images.append(image_resized)

    new_height,new_width=int(number4*height/100),int(number4*width/100) 
    image_resized=escaled_up_image_save(image_1,new_width,new_height,3)
    array_of_images.append(image_resized)
    return array_of_images

def escaled_up_image_save(image_1,new_width,new_height,iteration):
    dsize=(new_width,new_height)
    image_resized= cv2.resize(image_1, dsize)
    path='/Users/PC/Downloads/IA/2_corte/keyPoints_C2.Act_3/ImagenesEscaladas/'
    cv2.imwrite(path+'image'+str(iteration)+'.jpg',image_resized) 
    return image_resized







  