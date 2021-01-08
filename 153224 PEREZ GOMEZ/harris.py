from pylab import *
from numpy import *
from scipy.ndimage import filters
import cv2
import numpy

def image_gray(image_rgb):
    image_gray= cv2.cvtColor(image_rgb,cv2.COLOR_BGR2GRAY)
    return image_gray

def dataset(image_gray):
    image_gray= np.float32(image_gray)
    dataset= cv2.cornerHarris(image_gray,2,3,0.04)
    return dataset

def get_points(imagen,dst):
    coordinates_x_y=[]
    KeyPoints=[]
    pixels=[]
    height,width=dst.shape
    for y in range(0, height):
        for x in range(0, width):
            if dst.item(y, x) > 0.01 * dst.max():
                coords =tuple([x,y])
                coords=np.array(coords)
                coordinates_x_y.append(coords) 
                KeyPoints.append(cv2.KeyPoint(x,y,0))
                value=imagen[x][y].flatten()
                pixels.append(value)         
    return pixels,KeyPoints,coordinates_x_y       

def draw_points(imagen,coords):
    for i in range(len(coords)):
        x=coords[i,0]
        y=coords[i,1]
        cv2.circle(imagen, (x,y), 2,(0,0,0), cv2.FILLED, cv2.LINE_AA)
    return imagen

def matches(coordinates_x_y_1,coordinates_x_y_2,kp_transformed):
    list_x_y_1=[]
    list_x_y_2=[]
    for i in range(len(kp_transformed)):
        for j in range(len(coordinates_x_y_2)):
            x_transformed=kp_transformed[i,0]
            y_transformed=kp_transformed[i,1]
            x_normal=coordinates_x_y_2[j,0]
            y_normal=coordinates_x_y_2[j,1] 
            if(x_transformed==x_normal and y_transformed==y_normal):
                list_x_y_1.append(coordinates_x_y_1[i])
                list_x_y_2.append(coordinates_x_y_2[j])
    return list_x_y_1,list_x_y_2

def match(desc1,desc2,threshold=0.5):
    """For each corner point descriptor in the first image, 
    select its match to second image using normalized cross correlation. 
    n = len(desc1[0])"""
    # pair-wise distances
    d = -ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            ncc_value = sum(d1 * d2) / (n-1) 
            if ncc_value > threshold:
                d[i,j] = ncc_value
            
    ndx = argsort(-d)
    matchscores = ndx[:,0]
    
    return matchscores

#ROTACION 
#########################################################
def rotated_coords(points,M):
    points = np.array(points)
    ones = np.ones(shape=(len(points),1))
    points_ones = np.concatenate((points,ones), axis=1)
    transformed_pts = M.dot(points_ones.T).T
    return transformed_pts

def convert_to_coordinates(kp_1_transformed):
    new_coords_kp1_transformed=[]
    for i in range(len(kp_1_transformed)):
        x=kp_1_transformed[i,0]
        y=kp_1_transformed[i,1]
        coordinate_x=np.array(math.modf(x))
        coordinate_y=np.array(math.modf(y))
        new_x=round(coordinate_x[1])#int
        new_y=round(coordinate_y[1])#int
        coords =tuple([new_x,new_y])
        coords=np.array(coords)
        new_coords_kp1_transformed.append(coords)
    new_coords_kp1_transformed=np.array(new_coords_kp1_transformed)
    return new_coords_kp1_transformed

def rotation_matches(coordinates_x_y_1,coordinates_x_y_2,new_coords_kp1_transformed):
    list_x_y_1=[]
    list_x_y_2=[]
    for i in range(len(new_coords_kp1_transformed)):
        for j in range(len(coordinates_x_y_2)):
            x_transformed=new_coords_kp1_transformed[i,0]
            y_transformed=new_coords_kp1_transformed[i,1]
            x_rotated=coordinates_x_y_2[j,0]
            y_rotated=coordinates_x_y_2[j,1] 
            if(x_transformed==x_rotated and y_transformed==y_rotated):
                list_x_y_1.append(coordinates_x_y_1[i])
                list_x_y_2.append(coordinates_x_y_2[j])
    return list_x_y_1,list_x_y_2  

#DESPLAZAMIENTO  
############################################################
def kp_transform_deplaced_image(coordinates_x_y_1,x,y):
    coordinates_x_y_1=np.array(coordinates_x_y_1)
    kp_transformed=[]
    if(x>0 and y>0):
        for i in range(len(coordinates_x_y_1)):
            new_x=coordinates_x_y_1[i,0]
            new_y=coordinates_x_y_1[i,1]
            coords=tuple([new_x+x,new_y+y])
            coords=np.array(coords)
            kp_transformed.append(coords)
        return kp_transformed

    if(x<0 and y<0):
        for i in range(len(coordinates_x_y_1)):
            new_x=coordinates_x_y_1[i,0]
            new_y=coordinates_x_y_1[i,1]
            coords=tuple([new_x+x,new_y+y])
            coords=np.array(coords)
            kp_transformed.append(coords)
        return kp_transformed   

    if(x>0 and y<0):
        for i in range(len(coordinates_x_y_1)):
            new_x=coordinates_x_y_1[i,0]
            new_y=coordinates_x_y_1[i,1]
            coords=tuple([new_x+x,new_y+y])
            coords=np.array(coords)
            kp_transformed.append(coords)
        return kp_transformed 
    
    if(x<0 and y>0):
        for i in range(len(coordinates_x_y_1)):
            new_x=coordinates_x_y_1[i,0]
            new_y=coordinates_x_y_1[i,1]
            coords=tuple([new_x+x,new_y+y])
            coords=np.array(coords)
            kp_transformed.append(coords)
        return kp_transformed          

    if(x<0 and y==0):
        for i in range(len(coordinates_x_y_1)):
            new_x=coordinates_x_y_1[i,0]
            new_y=coordinates_x_y_1[i,1]
            coords=tuple([new_x+x,new_y])
            coords=np.array(coords)
            kp_transformed.append(coords)
        return kp_transformed

    if(x>0 and y==0):
        for i in range(len(coordinates_x_y_1)):
            new_x=coordinates_x_y_1[i,0]
            new_y=coordinates_x_y_1[i,1]
            coords=tuple([new_x+x,new_y])
            coords=np.array(coords)
            kp_transformed.append(coords)
        return kp_transformed

    if(x==0 and y>0):
        for i in range(len(coordinates_x_y_1)):
            new_x=coordinates_x_y_1[i,0]
            new_y=coordinates_x_y_1[i,1]
            coords=tuple([new_x,new_y+y])
            coords=np.array(coords)
            kp_transformed.append(coords)
        return kp_transformed

    if(x==0 and y<0):
        for i in range(len(coordinates_x_y_1)):
            new_x=coordinates_x_y_1[i,0]
            new_y=coordinates_x_y_1[i,1]
            new_y=new_y+y
            coords=tuple([new_x,new_y])
            coords=np.array(coords)
            kp_transformed.append(coords)
        return kp_transformed

###################################################
#ESCALADO
def transformed_kp_scaled(image_original,image_resized,coordinates_x_y_1,coordinates_x_y_2,scaled):
    height_original,width_original=image_original.shape[:2] 
    height_resized,width_resized=image_resized.shape[:2] 
    center_original_x,center_original_y=round(width_original/2),round(height_original/2)
    center_resized_x,center_resized_y=round(width_resized/ 2),round(height_resized/2)
    if(scaled>=200):
        numer_to_multiply=int(scaled/100)
        new_coordinates_x_y_1=[]
        new_coordinates_x_y_1=KeyPoints_Transformed(coordinates_x_y_1,coordinates_x_y_2,numer_to_multiply,center_original_x,center_original_y,center_resized_x,center_resized_y)
        return new_coordinates_x_y_1
    else:
        numer_to_multiply=scaled
        new_coordinates_x_y_1=[]
        new_coordinates_x_y_1=KeyPoints_Transformed(coordinates_x_y_1,coordinates_x_y_2,numer_to_multiply,center_original_x,center_original_y,center_resized_x,center_resized_y)
        return new_coordinates_x_y_1

def KeyPoints_Transformed(coordinates_x_y_1,coordinates_x_y_2,numer_to_multiply,center_original_x,center_original_y,center_resized_x,center_resized_y):
    new_coordinates_x_y_1=[]
    for i in range(len(coordinates_x_y_1)):
        x_kp=coordinates_x_y_1[i,0]-center_original_x
        y_kp=coordinates_x_y_1[i,1]-center_original_y
        x_kp=x_kp*numer_to_multiply
        y_kp=y_kp*numer_to_multiply
        if(x_kp>0 and y_kp>0):
            new_x_kp=center_resized_x+x_kp
            new_y_kp=center_resized_y+y_kp
            coords =tuple([new_x_kp,new_y_kp])
            coords=np.array(coords)
            new_coordinates_x_y_1.append(coords)
        if(x_kp<0 and y_kp<0):
            new_x_kp=center_resized_x-(x_kp*-1)
            new_y_kp=center_resized_y-(y_kp*-1)
            coords =tuple([new_x_kp,new_y_kp])
            coords=np.array(coords)
            new_coordinates_x_y_1.append(coords)
        if(x_kp>0 and y_kp<0):
            new_x_kp=center_resized_x+x_kp
            new_y_kp=center_resized_y-(y_kp*-1)
            coords =tuple([new_x_kp,new_y_kp])
            coords=np.array(coords)
            new_coordinates_x_y_1.append(coords)
        if(x_kp<0 and y_kp>0):
            new_x_kp=center_resized_x-(x_kp*-1)
            new_y_kp=center_resized_y+y_kp
            coords =tuple([new_x_kp,new_y_kp])
            coords=np.array(coords)
            new_coordinates_x_y_1.append(coords)    
    return new_coordinates_x_y_1

#########################################################

"""def compute_harris_response(im,sigma=3):#no la uso 
    #Compute the Harris corner detector response function 
    #for each pixel in a graylevel image. 
    print("sigma",sigma)
    # derivatives
    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
    
    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx,sigma)
    Wxy = filters.gaussian_filter(imx*imy,sigma)
    Wyy = filters.gaussian_filter(imy*imy,sigma)
    
    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    
    return Wdet / Wtr

def get_harris_points(harrisim,min_dist=10,threshold=0.01):
    #Return corners from a Harris response image
    #min_dist is the minimum number of pixels separating 
    #corners and image boundary.
    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t =(harrisim > corner_threshold) * 1
    # get coordinates of candidates
    coords = array(harrisim_t.nonzero()).T
    # ...and their values
    candidate_values = [harrisim[c[0],c[1]] for c in coords]
    # sort candidates
    index = argsort(candidate_values)
    # store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist), 
                        (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    
    return filtered_coords
    
def get_keyPoints(coordinates):
    keyPoints=[]
    for i in range(len(coordinates)):
        keypoint =cv2.KeyPoint(coordinates[i,0],coordinates[i,1],0)
        keyPoints.append(keypoint)
    
    return keyPoints 

def get_descriptors(image,filtered_coords,wid=5):
    #For each point return pixel values around the point
    #    using a neighbourhood of width 2*wid+1. (Assume points are 
    #    extracted with min_distance > wid). 
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0]-wid:coords[0]+wid+1,
                            coords[1]-wid:coords[1]+wid+1].flatten()
        desc.append(patch)

    return desc

def match(desc1,desc2,threshold=0.5):# 
    #For each corner point descriptor in the first image, 
    #    select its match to second image using
    #    normalized cross correlation. 
    n = len(desc1[0])
    
    # pair-wise distances
    d = -ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            ncc_value = sum(d1 * d2) / (n-1) 
            if ncc_value > threshold:
                d[i,j] = ncc_value
            
    ndx = argsort(-d)
    matchscores = ndx[:,0]
    
    return matchscores

def coordinates_KeyPoint(DataSet,image):#sin holgura
    threshold=0.1
    coordinates=numpy.where(DataSet>threshold*DataSet.max())*1
    coordinates=numpy.array(coordinates).T
    return coordinates

def descriptors(image,dst):
    coordinates=numpy.where(dst>0.01*dst.max())
    coordinates=numpy.array(coordinates).T
    descriptors=[]
    for i in range(len(coordinates)) :
        patch=image[coordinates[i,0],coordinates[i,1]].flatten()
        descriptors.append(patch) 
    descriptors=numpy.array(descriptors)
    return descriptors

def appendimages(im1,im2):#no lo uso 
    #Return a new image that appends the two images side-by-side. 
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]    
    rows2 = im2.shape[0]
    
    if rows1 < rows2:
        im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
    # if none of these cases they are equal, no filling needed.
    
    return concatenate((im1,im2), axis=1)
def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):#no lo uso 
    Show a figure with lines joining the accepted matches 
        input: im1,im2 (images as arrays), locs1,locs2 (feature locations), 
        matchscores (as output from 'match()'), 
        show_below (if images should be shown below matches). 
    im3 = appendimages(im1,im2)
    if show_below:
        im3 = vstack((im3,im3))
    

    cv2.imshow(im3)
   
    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'*')
    axis('off')"""


