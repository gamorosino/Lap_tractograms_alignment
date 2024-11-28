import sys
import os
import numpy as np

from alignment_as_lap_CLI import LAP, loadTrk, distance_corresponding, alignment_as_LAP,load_tractogram
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.distances import bundles_distances_mam, bundles_distances_mdf

def LAPD(T_moving,T_static,nb_points=16,distance_function = bundles_distances_mdf,clusters=False,average=True, metric = bundles_distances_mdf,swap=False):
        
        if nb_points is not None:
        
                T_moving=set_number_of_points(  T_moving , nb_points )
                T_static=set_number_of_points(  T_static, nb_points )	        
                T_moving=np.array(T_moving, dtype=object)
                T_static=np.array(T_static, dtype=object)
                       
        if clusters:
                
                T_moving=np.array(T_moving, dtype=object)
                T_static=np.array(T_static, dtype=object)
                
                k = 5000  # number of clusters, usually somewhat above sqrt(|T_A|) is optimal for efficiency.
                threshold_short_streamlines = 0.0  # Beware: discarding streamlines affects IDs
                # Additional internal parameters for mini-batch k-means, no need to change them, usually:
                b = 100
                t = 100   
                T_moving = np.array(T_moving, dtype=object)
                T_static = np.array(T_static, dtype=object)
                correspondence, distances = alignment_as_LAP(T_A=T_moving,
                                                        T_B=T_static,
                                                        k=k,
                                                        threshold_short_streamlines=threshold_short_streamlines,
                                                        b=b,
                                                        t=t,
                                                        distance_function=distance_function,swap=swap)
                distances = distance_corresponding(np.array(T_moving), np.array(T_static), correspondence, distance_function=metric,swap=swap)
        else:
 
                correspondence = LAP(T_moving,T_static)
                distances = distance_corresponding(np.array(T_moving), np.array(T_static), correspondence, distance_function=metric,swap=swap)
        if average:
                return np.mean(distances)
        else:
                return distances

if __name__ == '__main__':
    
        distance_function = bundles_distances_mdf  # bundles_distances_mdf is faster
        nb_points = 16
        T_A_filename = sys.argv[1] 
        T_B_filename = sys.argv[2] 
        try:
                clusters = bool(int(sys.argv[3] ))
        except:
                clusters = False
        
        if clusters:
                threshold_short_streamlines = 0.0  # Beware: discarding streamlines affects IDs
                T_moving = load_tractogram(T_A_filename,
                                                                        threshold_short_streamlines=threshold_short_streamlines,
                                                                        nb_points=nb_points)
                T_static = load_tractogram(T_B_filename,
                                                                        threshold_short_streamlines=threshold_short_streamlines,
                                nb_points=nb_points)  
                nb_points = None              
        else:     
                T_moving, _ , _ = loadTrk(T_A_filename)
                T_static, _ , _  = loadTrk(T_B_filename)
        #%% Resample Streamlines
        print("setting the same number of points...")
        #T_moving = np.array(T_moving)
        #T_static = np.array(T_static)
        distance = LAPD(T_moving,T_static,nb_points=nb_points,distance_function=distance_function, clusters=clusters)
        print(distance)
