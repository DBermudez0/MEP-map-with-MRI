def MEP_map_with_MRI():
    #=============================================================================
    #Implementation of Machine Learning Classification for Motor Evoked Potential
    #acquired throught Transcranial Magnetic Stimulation
    #=============================================================================

    
    # Modified for classification of MEP data obtain from neuronavigated TMS and
    # Mapping of contour classification into MRI by Dalton H Bermudez
   

    # added extra lines

    # Start code:
    # Type in the Python Command line the following lines:
    # >> import MEP_map_with_MRI as MEP_soft
    # >> MEP_soft.MEP_map_with_MRI()
    # make sure the code is in the same directory were you are running the Python shell
    
 
    import glob
    import numpy as np
    import pandas as pd
    import os
    import nibabel as nib
    from pylab import ginput
    import matplotlib.pyplot as plt
    import scipy as scp
    from matplotlib.colors import ListedColormap
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.svm import SVC
    from sklearn.gaussian_process.kernels import RBF
    from PIL import Image

    
    import warnings
    warnings.filterwarnings("ignore")
    correct_final = False
    count_6 = 0
    while(correct_final != True):
        def tellme(s):
            print(s)
            plt.title(s, fontsize=16)
            plt.draw()
        
        correct_1 = False    
        while (correct_1 != True): 
            view = input('Do you want to see MEP maps in 2D structures or 3D structure (type:2D or 3D): ')
            #print(view)
            if view == '2D':
                Path_MRI = str(input('Specify the Path to nii?:'))
                File_name = str(input('What is the name of file?:'))
              
                mri_filename = os.path.join(Path_MRI, File_name)
            
                mri = nib.load(mri_filename)
                mri_data = mri.get_data()
                mri_data_image = mri_data[:,350,:]
                mri_dims = mri_data_image.shape
                correct_1 = True
            elif view == '3D':
                Path_MRI = str(input('Specify the Path to 3D structure image?:'))
                File_name = str(input('What is the name of file?:'))
               
                mri_filename = os.path.join(Path_MRI, File_name)
                im = Image.open(mri_filename)
             
                correct_1 = True
            else:
                print("Invalid input")


        h = 0.1 # step size in the mesh

        # defines array
        np_array_1 = [] 
        np_array_2 = []
        np_array_3 = []
        np_array_4 = []
        MEP_1 = []
        MEP_2 = []
        MEP_3 = []
        MEP_4 = []
        data_1 = []
        data_2 = []
        df_1 = []
        data_1 = []
        X2 = []
        X1 = []
        MEP1_binary = []
        MEP2_binary = []
        np_MEP_hand, np_MEP_forearm = [], []
        figure = []
        categies = []

        Path_inquery = str(input('Do you want to specify a new Path to files?[Y or enter key to keep the previous]:'))
        if Path_inquery.lower() == 'y':
            count = 0
            Path = input('Specify the Path of all the CSV file:')
        allfile_in_dir = glob.glob(os.path.join(Path, '*.csv'))


        
        for i in range(0,len(allfile_in_dir)):
            print('[' + str((i + 1)) + '] ' + str(allfile_in_dir[i][len(Path):-4]))
         
            
            

        Patient_file = input('Specify the Patient file name to be processed:')
        count_fig = 0
        allfiles = glob.glob(os.path.join(Path, Patient_file + '.csv'))
      
        
        for k in range(0, len(allfiles)):
            categies.append(str(allfiles[k][len(Path):-4]))
        count_end = len(allfiles)
        size = len(allfiles)


        names = ["RBF SVM"]

        classifiers = [
            SVC(kernel = 'rbf',gamma='auto', C=0.025),#gamma influence the level of backgrand intensity and C in the weighting applied to each class 
            ]

        for num in range(0, size):
            
                df_1 = pd.read_csv(allfiles[num])
                data_1 = df_1.as_matrix()

                data_1_hand = data_1[:,0:4]
                s_1 = len(data_1_hand)


                for j in range(0, s_1):
                        if np.all(data_1_hand[j,:] != '-'):
                                np_array_1.append(data_1_hand[j,:])
                                MEP_1= np.array(np_array_1)
                                X1 = MEP_1.astype(np.float)
                            
                max_MEP1 = max([float(k) for k in X1[:,3]])
                binary_1 = X1[:,3]/max_MEP1
                binary_1 = binary_1[~np.isnan(binary_1)]
                coordinates_1 = X1[:,[0,2]]
                coordinates_1 = coordinates_1[~np.isnan(coordinates_1)]
                coordinates_1 = coordinates_1.reshape((int(len(coordinates_1)/2), 2))
                MEP1_binary = (coordinates_1, binary_1)
                np_MEP_hand.append(MEP1_binary)
                min_MEP1 = round(min([float(k) for k in binary_1]),2)
                            
                correct_2 = False    
                while (correct_2 != True): 
                    Thresh_hand = float(input('Input a Threshold value between ' + str(min_MEP1) +' and 1 (for Hand): '))
                    if (Thresh_hand > min_MEP1 and Thresh_hand < 1):
                        for index, item in enumerate(binary_1):
                            if item >= Thresh_hand:
                                binary_1[index] = 1
                            else:
                                binary_1[index] = 0
                        correct_2 = True
                    else:
                        print('Input Threshold must be between ' + str(min_MEP1) +' and 1')
                    
           
                data_1_forearm = np.hstack((data_1[:,0:3],data_1[:,4:5]))
                s_3 = len(data_1_forearm)
           
                for j in range(0, s_3):
                        if np.all(data_1_forearm[j,:] != '-'):
                                np_array_3.append(data_1_forearm[j,:])
                                MEP_3= np.array(np_array_3)
                                X2 = MEP_3.astype(np.float)
                    
                             
                max_MEP2 = max([float(k) for k in X2[:,3]])
                binary_2 = X2[:,3]/max_MEP2
                binary_2 = binary_2[~np.isnan(binary_2)]
                coordinates_2 = X2[:,[0,2]]
                coordinates_2 = coordinates_2[~np.isnan(coordinates_2)]
                coordinates_2 = coordinates_2.reshape((int(len(coordinates_2)/2), 2))
                MEP2_binary = (coordinates_2, binary_2)
                np_MEP_forearm.append(MEP2_binary)
                min_MEP2 = round(min([float(k) for k in binary_1]), 2)
            
                correct_3 = False    
                while (correct_3 != True): 
                    Thresh_forearm = float(input('Input a Threshold value between ' + str(min_MEP2) +' and 1 (for Forearm): '))
                    if (Thresh_forearm > min_MEP2 and Thresh_forearm < 1):
                        for index, item in enumerate(binary_2):
                            if item >= Thresh_forearm:
                                binary_2[index] = 1
                            else:
                                binary_2[index] = 0
                        correct_3 = True
                    else:
                        print('Input Threshold must be between ' + str(min_MEP2) +' and 1')
          
        for num in range(0, size):
            datasets = [np_MEP_hand[num],
                    np_MEP_forearm[num]]


            i =1
            figure = plt.figure(figsize=(20, 9))
            figure.suptitle(str(categies[num]), fontsize=14, fontweight='bold')
            plt.grid(False)  
            # iterate over datasets
            for ds_cnt, ds in enumerate(datasets):
                # preprocess dataset, split into training and test part
                X, y = ds
                X_1 = X
                X = StandardScaler().fit_transform(X)
                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=.5, random_state=42)


                x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
                y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))


                # just plot the dataset first
                cm = plt.cm.bwr #RdBu
                cm_bright = ListedColormap(['#0000FF', '#FF0000'])
                ax = plt.subplot(1, len(classifiers)+1, i)

                # iterate over classifiers
                for name, clf in zip(names, classifiers):
                    #if num ==
                    ax = plt.subplot(1 , len(classifiers) + 1, i)# chaged len(dataset) to num
                    clf.fit(X_train, y_train)
                    score = clf.score(X_test, y_test)

                    # Plot the decision boundary. For that, we will assign a color to each
                    # point in the mesh [x_min, x_max] [y_min, y_max].
                    if hasattr(clf, "decision_function"):
                        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                    else:
                        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                    

                    # Put the result into a color plot
                    Z = Z.reshape(xx.shape)
               

                    if view == '2D':
                        mri_data_image = scp.misc.imresize(mri_data_image,(1800, 1600), interp = 'bilinear')
                    elif view == '3D':
                        if count_fig == 0: 
                            im.show()
                            print('Orientation of 3D structures of the brain needs to lie vertically')
                            angle = float(input('What angle you want to tilt the image to 3D Brain structures verticle? (- angle is clockwise):'))
                            count_fig += 1
                        out = im.rotate(angle)
                        mri_data_image = scp.misc.imresize(out,(1800, 1600), interp = 'bilinear')
                        
                    ax.imshow(mri_data_image)
                   
                 
                    tellme('Select one point in the area of TMS stimulation')
                    pts = np.asarray(ginput(1))
                    Z_scale = scp.misc.imresize(Z,(40, 40), interp = 'bilinear')
             
                    xx_scale = scp.misc.imresize(xx, (40,40), interp = 'bilinear')
                    yy_scale = scp.misc.imresize(yy, (40, 40), interp = 'bilinear')

                    ax.contourf(xx_scale+pts[:,0], yy_scale + pts[:,1], Z_scale, cmap=cm, alpha=0.7)

                    ax.set_xticks(())
                    ax.set_yticks(())
                    if ds_cnt == 0:
                            ax.set_title("MEP Hand")
                    if ds_cnt == 1:
                            ax.set_title("MEP Forearm")
                    ax.text(xx_scale.max()+pts[:,0]+ - .3, yy_scale.max()+ pts[:,1]+ .3, ('%.2f' % score).lstrip('0'),
                            size=10, horizontalalignment='right')
                    i += 1
                    
        plt.tight_layout()
        plt.show()
        count_6 += num
        terminate = str(input('Do you want to quit? (Q or press enter key to continue):'))
        if (count_6 == count_end):
            print('You have processed all the pateint cases in these folder')
            correct_final = True
        elif (terminate.lower() == 'q'):
            correct_final = True
   
    if __name__ == "___MEP_map_with_MRI___":
        MEP_map_with_MRI()










