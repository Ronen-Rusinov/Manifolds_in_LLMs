### PIPELINE ROADMAP ###
This document details the process of running the iddferent scripts of the project 
In sequence, in order to produce the necessary results for the project.
Below is a diagram of the total pipeline:
                                                    
    (*) Pull data                                   --------------------------------- 
    ||  ----------                                  |Numbers in [brackets] designate|
    ||  Use scripts/pull_data.py                    |default parameters, but are    |
    ||  to pull the data from the                   |subject to taste               |
    ||  associated google drive                     ---------------------------------
    ||  Into /data/activations_data/
    ||
    ##==============================================##==============================##
    ||                                              ||                              ||
    (*) Clustering                                  (*) Produce ball tree           (*) View activation norms
    ||  ------------                                ||-------------------               ---------------------
    || Use scripts/minibatch-kmeans                 ||Use                               Use experiments/activation_norms.py
    ||                                              ||scripts/produce_balltree.py       to produce a histogram of 
    || To produce [200] clusters on the layer       ||to produce a balltree             activation norms in both main layers
    ||  18  activations and save them to            ||for the entirety of the               
    ||  results/minibatch_kmeans/centroids.npy      ||data                                                  
    ||                                              ||
    ||                                             //
    ||                                            //
    ||                                           //
    ||                                          //
    ||                                         //
    ||                                        //
    ||                                       //
    ||                                     ...
    ||
    || ...
    ||//
    (*) Obtain neighbors
    ||  -----------------
    || Use obtain_K_nearest_to-centroids.py
    || to obtain the nearest [10,000]
    


