import numpy as np
import scipy.ndimage as ndi

def seedGrowingSeg(I,sigma,seed_x,seed_y,thresh,n_rows,n_cols,slice_num):  
    # seedGrowingSeg segments a 2D image using region growing algorithm
    
    # FUNCTION DESCRIPTION: 
    #   1) Receives 2D slice and performs pre-processing Gaussian blurring
    #   2) Iteratively finds all non-segmented pixels connected to the current 
    #      segmented region
    #   3) Compares similarity of connected pixels to mean of segmented region
    #   4) If smallest pixel intensity difference is below threshold value
    #      then this pixel is added to segmentation
    #   5) If smallest pixel instensity difference is greater than
    #      threshold value, or if maximum number of iterations has beed 
    #      exceeded, then algorithm has converged 

    # INPUTS:
    #   I - 2D image normalised between 0 and 1 - float
    #   sigma - width of Gaussian kernel for blurring - float
    #   seed_x - x coordinate of initial seed point - int
    #   seed_y - y coordinate of initial seed point - int
    #   thresh - threshold for convergence of segmentation - float
    #   n_rows - number of rows in I - int
    #   n_cols - number of columns in I - int
    #   slice_num - slice number of current slice in whole dataset - int
    
    # OUTPUTS:
    #   Im_blur - output of 2D Gaussian blurring on I - float
    #   seg - logical segmentation output - bool
    #   region_size - number of pixels in segmented region - int
    
    # FUNCTION DEPENDENCIES:
    #   numpy
    #   scipy.ndimage
    
    # AUTHOR:
    #   Edward James, June 2020
    
    #%% pre processing - 2D gaussian blurring to smooth out noise
    Im_blur = ndi.gaussian_filter(I,sigma,order=0,mode='reflect') 
    
    #%% preallocate memory for segmentation output, also specify maximum 
    # number of iterations as 10% of image area
    seg = np.zeros([n_rows,n_cols],bool)
    max_num_iterations = round(n_rows*n_cols/10)
    iterations = np.arange(1,max_num_iterations)
        
    # preallocate memory for coords of segmentation logical labels
    all_x_coords = np.zeros(max_num_iterations,int)
    all_y_coords = np.zeros(max_num_iterations,int)
    
    # initialise first iteration 
    seg[seed_y,seed_x] = 1
    region_mean        = Im_blur[seed_y,seed_x]
    all_x_coords[0]    = seed_x
    all_y_coords[0]    = seed_y
    
    #%% define 2D (i.e. in-plane) 4 degree connectivity of a given pixel centred at (0,0)
    # pixels are connected if their edges touch
    kernel = np.zeros([4,2],int)
    kernel[0,0] =  1
    kernel[1,1] = -1
    kernel[2,0] = -1
    kernel[3,1] =  1
    connectivity = np.arange(4)
    
    #%% start iterative region growing loop 
    
    for num_iter in iterations:
        # preallocate temporary matrix of all unassigned connected pixels
        connected_pix = np.zeros([n_rows,n_cols])
        # acquire coordinates of all connected pixels 
        for i in connectivity:
            conn_y_coord = all_y_coords[0:num_iter] + kernel[i,0]
            conn_x_coord = all_x_coords[0:num_iter] + kernel[i,1]
            # loop through each of these connected pixels and add to temporary matrix 
            n_coords = np.arange(len(conn_x_coord))
            for j in n_coords:
               # if pixel address is contained in image 
               if 0 <= conn_y_coord[j] and conn_y_coord[j] < n_rows \
                   and 0 <= conn_x_coord[j] and conn_x_coord[j] < n_cols:
                       # if not already part of current segmentation
                       if seg[conn_y_coord[j],conn_x_coord[j]] != 1:
                           connected_pix[conn_y_coord[j],conn_x_coord[j]] = 1
        # multiply blurred image by this logical mask
        connected_pix_intensities = Im_blur*connected_pix
        # find the pixel which has the smallest absolute intensity difference to the
        # current region mean
        sim_metric_all = np.reshape(abs(region_mean-connected_pix_intensities),n_cols*n_rows)
        # calculate smallest current similarity metric and location of respective pixel
        # in flattened array
        sim_metric = min(sim_metric_all)
        ind = np.argmin(sim_metric_all)
        # if this absolute intensity difference is smaller than threshold then add 
        # this pixel to segmentation region and update region mean
        if sim_metric < thresh:
            # convert this 1D idx to 2D coords
            [new_y_idx,new_x_idx] = np.unravel_index(ind,(n_rows,n_cols))
            # and add to all_coords
            all_x_coords[num_iter] = new_x_idx
            all_y_coords[num_iter] = new_y_idx
            # and add to segmentation
            seg[new_y_idx,new_x_idx] = 1
            # update region mean
            region = Im_blur*seg
            region_mask = region[region != 0]
            region_mean = np.mean(region_mask)
        else:
            print("Slice {} converged".format(slice_num+1))
            # break out of outer for loop
            break
    
    # calculate total number of pixels in segmentation
    region_size = sum(sum(region != 0))
    
    return [Im_blur,seg,region_size]