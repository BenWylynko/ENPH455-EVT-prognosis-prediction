
import nibabel as nib
import nrrd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import scipy.ndimage as ndimage
import gc

DATA_BASE = r'Z:\EVT Project Data\NiftII\Complete'
SAVE_BASE = r'C:\Users\Ben\Downloads\data'

def load_niftii(fpath, shape_std=True):
    """
    Load a single .nii file
    Data coordinates:
    [from fore-brain to rear brain in anterior view, flipped across vertical relative to 3d slicer
    from top to bottom in superior view, flipped across vertical relative to 3d slicer
    in lateral view] inverted 90deg both ways relative to 3d slicer

    Only gives 1 channel, is this ok?
    """
    img = nib.load(fpath)
    header = img.header
    data = img.get_fdata()
    if shape_std:
        #check if padding is needed
        if data.shape[2] < 512:
            #pad to max possible shape (512, 512, 519) with value -1024 (background)
            padding = 519 - data.shape[2]
            if padding % 2 == 0: #even value to pad
                p = padding // 2
                data = np.pad(data, ((0, 0), (0, 0), (p, p)), constant_values=-1024)
            else: #odd value to pad
                p = padding // 2
                data = np.pad(data, ((0, 0), (0, 0), (p, p + 1)), constant_values=-1024)
    return data

def save_niftii(data, fpath):
    img = nib.Nifti1Image(data, np.eye(4))
    print(f"Saving volume to {fpath}")
    nib.save(img, fpath)

def load_stl(fpath):
    """Load stl file + convert to a raster format"""

    

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
    #try loading in chunks to avoid memory error
    plist = [Path(DATA_BASE, id_name) for id_name in ids_names]
    return np.array([load_niftii(p) for p in plist])

def load_all_segmentations(id_list):
    """
    Load multiple NRRD files based on patient ID list
    """
    plist = [Path(DATA_BASE, id, 'segmentation.nrrd') for id in id_list]
    return np.array([load_nrrd(p) for p in plist])

def normalize(volume, min=-1000, max=500):
    """Normalize the volume"""
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("int16")
    return volume

def resize_volume(img, volume=256):
    """Resize across z-axis"""
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / volume
    width = current_width / volume
    height = current_height / volume
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    print(type(img))
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def save_multiple(ids, volume=256, start_id=0, end_id=-1):
    """Process and save multiple NIFTII files"""
    for f in ids[start_id:end_id]:
        print(f"Processing {f}")
        #can be niftii or nrrd
        print(f.suffix)
        if f.suffix == '.nrrd':
            data = load_nrrd(Path(DATA_BASE, f))
        else:
            data = load_niftii(Path(DATA_BASE, f))
        print(data.shape)
        data = normalize(data)
        vol = resize_volume(data, volume=volume)
        print(f.stem)
        #define save path
        if f.suffix is not ('.nrrd' or 'nii'):
            savepath = Path(SAVE_BASE, f'{volume}_{volume}', f'{f.parents[0].stem}_{f.stem}.nii')
        else:
            savepath = Path(SAVE_BASE, f'{volume}_{volume}', f'{f.parents[0].stem}_{f.stem}')
        save_niftii(vol, savepath)
        print(f"Done processing {f}\n")

        #does garbage collection after each one prevent my memory error?
        n_unreachable = gc.collect()
        print(f"{n_unreachable} unreachable objects for garbage collection")

if __name__ == "__main__":
    #test loading
    niipath = Path(r'Z:\EVT Project Data\NiftII\Complete\0732825\601 Sagittal Head - imageOrientationPatient 1.nii')
    nrrdpath = r'Z:\EVT Project Data\NiftII\Complete\0732825\segmentation.nrrd'


    ids = [Path('0157754', '7 Post Contrast_2.nii'), 
    Path('0174960', '11 Post Contrast.nii'), 
    Path('0337411', '601 Axial125x06 - imageOrientationPatient 1_11.nii'),
    Path('0357680', '4 Axial 1.25 x 1.25_1.nii'), 
    Path('0444750', '604 Sagittal MIP 5x2 - imageType DERIVED-SECONDARY-REFORMATTED-MIP_15.nii'), 
    Path('0497694', '602 Sagittal AR20 2x2 - imageOrientationPatient 1_4.nii'), 
    Path('0513431', '8 Post Contrast.nii'),  
    Path('0673127', '604 Sagittal MIP 5x2 - imageType DERIVED-SECONDARY-REFORMATTED-MIP_11.nii'),  
    Path('0702581', '12 Axial 125 Std.nii'),
    Path('0732825', '602 Coronal Head - imageType DERIVED-SECONDARY-REFORMATTED-AVERAGE.nii'), 
    Path('0778928', '602 Sagittal AR20 2x2 - imageOrientationPatient 1_8.nii'), 
    Path('0783417', '10 Post Contrast_1.nii'), 
    Path('0786080', '604 Sagittal MIP 5x2 - imageType DERIVED-SECONDARY-REFORMATTED-MIP_14.nii'), 
    Path('0840732', '5 Post Contrast.nii'), 
    Path('0863628', '10 Post Contrast_3.nii'), 
    Path('0869517', '7 Post Contrast_1.nii'), 
    Path('1069211', '7 Post Contrast_4.nii'), 
    Path('1175412', '10 Post Contrast_2.nii'), 
    Path('1302751', '7 Post Contrast_1.nii'), 
    Path('1305527', '603 Coronal MIP 5x2 - imageType DERIVED-SECONDARY-REFORMATTED-MIP_7.nii'), 
    Path('1310557', '601 Axial125x06 - imageOrientationPatient 1_8.nii'), 
]

    #save both 256 and 128 sized images
    save_multiple(ids, volume=128, start_id=5, end_id=6) #ind 5 brekas
    #save_multiple(ids, volume=256, start_id=6)


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
    """
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
    """
