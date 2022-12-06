import pandas as pd
import numpy as np
from scipy import misc
import cv2
import SimpleITK as sitk
import gui
import registration_gui as rgui


def FeaturesExtractor(model,img_data,i,j,tile_size):
    """model is the DCNN-based HF extractor;
    img_data is the H&E image;
    i,j are tile indice;
    tile_size is the size of a H&E image tile
    """
    from tensorflow.keras.applications.densenet import preprocess_input
    from tensorflow.keras.preprocessing import image
    im_roi_window=img_data[i*tile_size:(i+1)*tile_size,
                     j*tile_size:(j+1)*tile_size]
    img = cv2.resize(im_roi_window, (0, 0), fx=(224/100), fy=(224/100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # get the features 
    pool_features =model.predict(x)    
    return pool_features 




def HistomorphologicalFeatureConstructor(model,heImage,mask,tile_size):
    """
    calculate the HF spectra
    """
    
    import tensorflow as tf
    nRows=int(heImage.shape[0]/100)
    nCols=int(heImage.shape[1]/100)
    positionList=[]
    firstTime=True
    for i in range(nRows):
        for j in range(nCols):
            print(i,j)
            if mask[i,j]>0:
                    positionList.append((i,j))
                    if firstTime:
                        featuresData=FeaturesExtractor(model,heImage,i,j,tile_size)
                        firstTime=False
                    else:
                        featuresData=np.concatenate((featuresData,
                                                  FeaturesExtractor(model,heImage,i,j,
                                                  tile_size)),
                                                  axis=0)
    featuresDataPooled=tf.keras.layers.GlobalAveragePooling2D()(featuresData[:,:,:,:])
    sess = tf.compat.v1.Session()
    with sess.as_default():
         featuresNumpy=featuresDataPooled.numpy()
    return featuresNumpy
            

    
def TileTissueMask(heMask,tile_size,threshold=0.95):
    """
    remove H&E image tiles that belong to the background 
    """
    
    positionList=[]
    mask=np.zeros((int(heMask.shape[0]/tile_size),int(heMask.shape[1]/tile_size)))
    n_rows=int(heMask.shape[0]/tile_size)
    n_cols=int(heMask.shape[1]/tile_size)
    for i in range(n_rows):
        for j in range(n_cols):
            #print(i,j)
            if ((np.sum(heMask[i*int(tile_size):(i+1)*int(tile_size),
                                j*int(tile_size):(j+1)*int(tile_size)],
                        axis=(0,1))/(tile_size*tile_size))>threshold):
                mask[i,j]=1
                positionList.append((i,j))   
    return mask,positionList
                    
                    
def GenerateNMFScoreMapFromDataMatrix(HFDataMatrixScaled,tilePositions,
                                      tileMask,idx_component):

#Generate NMF score maps. 
#HFDataMatrixScaled is the 2D matrix of HF spectra.

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import NMF
    tilePositionsNp = np.array(tilePositions)
    dim_reducer=NMF(n_component)
    result = dim_reducer.fit_transform(HFDataMatrixScaled)
    
    NMFScoreMapOfHFData=np.zeros((tileMask.shape[0],tileMask.shape[1]))
    for i in range(len(tilePositions)):
         NMFScoreMapOfHFData[tilePositionsNp[i,0],tilePositionsNp[i,1]]=result[i,idx_component]
    
    
    return NMFScoreMapOfHFData   

def MsiDataCubeToDataMatrix(msiDataCube,msiMask):
    """
    reshape 3D MSI data cube to  2D matrix, only retain mass spectra
    of foreground pixels
    
    """
    positionList=[]
    msiDataList=[]
    n_rows=msiDataCube.shape[0]
    n_cols=msiDataCube.shape[1]
    
    for i in range(n_rows):
        for j in range(n_cols):
            print(i,j)
            if msiMask[i,j]>0:
                    positionList.append((i,j))
                    msiDataList.append(msiDataCube[i,j])
    return   msiDataList,  positionList                    
                    
                    
def Registration(fixed_np,moving_np,solutionAccuracy=1e-4):    
    # fixed image is the NMF score map of MSI data 
    # moving image is the NMF score map of HF data

    #from downloaddata import fetch_data as fdata

    fixed_image =  sitk.GetImageFromArray(fixed_np,isVector=False)
    moving_image = sitk.GetImageFromArray(moving_np,isVector=False)
   # fixed_image_mask=(fixed_np>0)
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                          moving_image,                                                            sitk.AffineTransform(fixed_image.GetDimension()),                                                           sitk.CenteredTransformInitializerFilter.GEOMETRY
                                                          )
    gui.RegistrationPointDataAquisition(fixed_image, moving_image, figure_size=(8,4), known_transformation=initial_transform,);
    affine_transform=AffineRegistration(fixed_image,moving_image,initial_transform)   
    from ipywidgets import interact, fixed
    from IPython.display import clear_output
    
    moving_resampled = sitk.Resample(moving_image, fixed_image, affine_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    bspline_transform= BsplineRegistration(fixed_image,moving_resampled,solutionAccuracy)
    moving_resampled_nonrigid = sitk.Resample(moving_resampled, fixed_image,  bspline_transform,  sitk.sitkLinear, 0.0, moving_resampled.GetPixelID())
    interact(display_images_with_alpha, image_z=(0),alpha=(0.0,1.0), fixed = fixed(fixed_image), moving=fixed(moving_resampled_nonrigid));
    return affine_transform,bspline_transform     
    
    

def AffineRegistration(fixed_image,moving_image,initial_transform):
    # linear registration
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric settings.
    #registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=10)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(1)
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1e-3, numberOfIterations=200, convergenceMinimumValue=1e-7, convergenceWindowSize=20)
    #registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    # Connect all of the observers so that we can perform plotting during registration.
    registration_method.AddCommand(sitk.sitkStartEvent, rgui.start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, rgui.end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, rgui.update_multires_iterations) 
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rgui.plot_values(registration_method))
    
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    # Always check the reason optimization terminated.
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    return final_transform
    

import matplotlib.pyplot as plt

from ipywidgets import interact, fixed
from IPython.display import clear_output

# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).
def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1,2,figsize=(10,8))
    
    # Draw the fixed image in the first subplot.
    plt.subplot(1,2,1)
    plt.imshow(fixed_npa[fixed_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('fixed image')
    plt.axis('off')
    
    # Draw the moving image in the second subplot.
    plt.subplot(1,2,2)
    plt.imshow(moving_npa[moving_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('moving image')
    plt.axis('off')
    
    plt.show()

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space. 
def display_images_with_alpha(image_z, alpha, fixed, moving):
    try:
        img = (1.0 - alpha)*fixed[:,:,image_z] + alpha*moving[:,:,image_z] 
    except:
        img = (1.0 - alpha)*fixed[:,:] + alpha*moving[:,:] 
    plt.imshow(sitk.GetArrayViewFromImage(img),cmap=plt.cm.Greys_r);
    plt.axis('off')
    plt.show()
    
# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))        
    
                        
def BsplineRegistration(fixed_image,moving_resampled,solutionAccuracy=1e-4):
    # nonlinear registration                                  
                                      
    # Define a simple callback which allows us to monitor registration progress.
    def iteration_callback(filter):
        print('\r{0:.2f}'.format(filter.GetMetricValue()), end='')
    
    registration_method = sitk.ImageRegistrationMethod()
        
    
    # The starting mesh size will be 1/4 of the original, it will be refined by 
    # the multi-resolution framework.
    mesh_size = [1,1]
    
    initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, 
                                                         transformDomainMeshSize = mesh_size, order=3)    
    # Instead of the standard SetInitialTransform we use the BSpline specific method which also
    # accepts the scaleFactors parameter to refine the BSpline mesh. In this case we start with 
    # the given mesh_size at the highest pyramid level then we double it in the next lower level and
    # in the full resolution image we use a mesh that is four times the original size.
    registration_method.SetInitialTransformAsBSpline(initial_transform,
                                                     inPlace=False,
                                                     scaleFactors=[1,2,4,8])
    
    #registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(1)
    #registration_method.SetMetricFixedMask(fixed_image_mask)
        
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [8,4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4,2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=solutionAccuracy, numberOfIterations=2000, deltaConvergenceTolerance=1e-3)
    
    registration_method.AddCommand(sitk.sitkStartEvent, rgui.start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, rgui.end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, rgui.update_multires_iterations) 
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rgui.plot_values(registration_method))
    
    final_transformation = registration_method.Execute(fixed_image, moving_resampled)
    print('\nOptimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    return final_transformation                    

                
                    
def bestCks(labels_1,labels_2,k):
    """
    calculate CKS for all possible permutations of the segmentation labels
    align the labels by maximizing CKS 
    """  
    from sklearn import metrics
    import itertools
    permutation_list=list(itertools.permutations(list(range(k))))
    permutation_dict={}
    for permutation in permutation_list:
        temp_labels=[]
        for i in labels_2:
            temp_labels.append(permutation[i])
        permutation_dict[metrics.cohen_kappa_score(labels_1, temp_labels)]=temp_labels
    return (max(permutation_dict.keys()), permutation_dict[max(permutation_dict.keys())])                        
                    
                    
                    
def cksCalculation(n_clusters_msi,n_clusters_morph,msi_features_np,morph_features_np): 
    #calculate CKS values for different combination of #Clusters                                  
    
   from sklearn.cluster import SpectralClustering
   clustering_msi = SpectralClustering(n_clusters=n_clusters_msi,affinity='cosine',n_components=5,
        assign_labels='kmeans',random_state=42).fit_predict(msi_features_np)
   clustering_morph = SpectralClustering(n_clusters=n_clusters_morph,n_components=5,
                                        affinity='nearest_neighbors',
        assign_labels='kmeans',random_state=21).fit_predict(morph_features_np)
   return bestCks(clustering_msi,clustering_morph,max([n_clusters_msi,n_clusters_morph]))[0]                    
                    
                    
                    
                    
                    
                    
                    
                    