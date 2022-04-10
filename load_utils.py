
import nibabel as nib
import nrrd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

DATA_BASE = r'Z:\EVT Project Data\NiftII\Complete'

def load_niftii(fpath):
    """
    Load a single .nii file
    Data coordinates:
    [from fore-brain to rear brain in anterior view, flipped across vertical relative to 3d slicer
    from top to bottom in superior view, flipped across vertical relative to 3d slicer
    in lateral view] inverted 90deg both ways relative to 3d slicer
    """
    img = nib.load(fpath)
    data = img.get_fdata()
    return data

def load_nrrd(fpath):
    """
    Load a single .nrrd file
    Data coordinates: TODO
    """
    data, header = nrrd.read(fpath)
    return data

def load_all_nii(ids_names):
    """
    Load multiple NII files, given list of format ["id/fname", ...]
    """
    plist = [Path(DATA_BASE, id_name) for id_name in ids_names]
    return np.array([load_niftii(p) for p in plist])

def load_all_nrrd(id_list):
    """
    Load multiple NRRD files based on patient ID list
    """
    plist = [Path(DATA_BASE, id, 'segmentation.nrrd') for id in id_list]
    return np.array([load_nrrd(p) for p in plist])


if __name__ == "__main__":
    #test loading
    niipath = r'Z:\EVT Project Data\NiftII\Complete\0732825\601 Sagittal Head - imageOrientationPatient 1.nii'
    nrrdpath = r'Z:\EVT Project Data\NiftII\Complete\0732825\segmentation.nrrd'

    niidata = load_niftii(niipath)
    nrrddata = load_nrrd(nrrdpath)

    print(f"Nii shape: {niidata.shape}")
    print(f"Nrrd shape: {nrrddata.shape}")

    #check data coordinates in nii file
    """
    #3rd dimension (lateral view)
    f, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(niidata[:, 50, :], aspect='auto')
    axs[1, 0].imshow(niidata[:, 100, :], aspect='auto')
    axs[0, 1].imshow(niidata[:, 150, :], aspect='auto')
    axs[1, 1].imshow(niidata[:, 200, :], aspect='auto')
    plt.show()

    #3rd dimension (lateral view)
    f, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(niidata[:, :, 10], aspect='auto')
    axs[1, 0].imshow(niidata[:, :, 20], aspect='auto')
    axs[0, 1].imshow(niidata[:, :, 30], aspect='auto')
    axs[1, 1].imshow(niidata[:, :, 40], aspect='auto')
    plt.show()


    #plot slices (nii)
    n_ims = niidata.shape[0] // 50 #10
    f, axs = plt.subplots(5, 2)
    for i in range(2):
        for j in range(5):
            ind = i * 250 + j * 50
            print(ind)
            axs[j, i].imshow(niidata[ind, :, :], aspect='auto')
            axs[j, i].set_title(f"{j}, {i}")
    plt.show()
    """


    #check data coordinates in nrrd file

    #3rd dimension (lateral view)
    f, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(nrrddata[:, 50, :], aspect='auto')
    axs[1, 0].imshow(nrrddata[:, 100, :], aspect='auto')
    axs[0, 1].imshow(nrrddata[:, 150, :], aspect='auto')
    axs[1, 1].imshow(nrrddata[:, 200, :], aspect='auto')
    plt.show()

    #3rd dimension (lateral view)
    f, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(nrrddata[:, :, 10], aspect='auto')
    axs[1, 0].imshow(nrrddata[:, :, 20], aspect='auto')
    axs[0, 1].imshow(nrrddata[:, :, 30], aspect='auto')
    axs[1, 1].imshow(nrrddata[:, :, 40], aspect='auto')
    plt.show()

    #plot slices (nrrd)
    n_ims = nrrddata.shape[0] // 10 #10
    f, axs = plt.subplots(5, 2)
    for i in range(2):
        for j in range(5):
            ind = i * 50 + j * 10
            print(ind)
            axs[j, i].imshow(nrrddata[ind, :, :], aspect='auto')
            axs[j, i].set_title(f"{j}, {i}")
    plt.show()


    #test loading a batch of nii files
    ids_names = [Path('0157754', '7 Post Contrast_2.nii'), Path('0357680', '4 Axial 1.25 x 1.25_1.nii'), Path('0513431', '8 Post Contrast.nii')]
    print(load_all_nii(ids_names))

    #test loading a batch of nrrd files
    id_list = ['0157754', '0357680', '0513431', '0732825', '0778928', '0783417', '0786080', '0863628', '0869517', '1069211', '1302751']
    #print(load_all_nrrd(id_list))


