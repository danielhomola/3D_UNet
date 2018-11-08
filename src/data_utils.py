import re
import pydicom
import nrrd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import display, clear_output

class Dataset(object):
    """
    Object to collect all patients within the train/test set.
    """
    def __init__(self, scan_files, seg_files):
        self.scan_files = scan_files
        self.seg_files = seg_files
        self.patients = self.build_dataset()
        self.patient_ids = list(self.patients.keys())
        
        
    def build_dataset(self):
        patients = dict()
        
        # read in all scans as a list - one file for each scan
        scans = [pydicom.dcmread(scan) for scan in self.scan_files]
        
        # read in all the segmentation files - one file for each patient
        segs = {seg.split('/')[-1]: nrrd.read(seg)[0] for seg in self.seg_files}
        
        # build dict of patient objects
        for i, scan in enumerate(scans):
            if scan.PatientID not in patients:
                # unfortunately, the PatientID cannot be trusted as sometimes 
                # it doesn't correspond to the appropriate nddr file, so we use 
                # regex to extract the patient_id from the folder of the DICOM file
                patient_folder = re.search(r'\/(Prostate[a-zA-Z0-9\-]+)\/', self.scan_files[i]).group(1)
                seg_file = segs['%s.nrrd' % patient_folder]
                patients[scan.PatientID] = Patient(scan=scan, seg=seg_file)
            else:
                patients[scan.PatientID].add_scan(scan=scan)
        
        # sort scans within each patient
        for patient in patients.keys():
            patients[patient].order_scans()
        
        return patients 

        
class Patient(object):
    """
    Basic object to store all slices of a patient in the study.
    """
    def __init__(self, scan, seg):
        self.scans = list()
        self.seg = seg
        self._instance_nums = list()
        self.thicknesses = set()
        self.manufacturers = set()
        self.add_scan(scan)
        
    def add_scan(self, scan):
        self.scans.append(scan.pixel_array)
        self._instance_nums.append(int(scan.InstanceNumber))
        self.thicknesses.add(int(scan.SliceThickness))
        self.manufacturers.add(scan.Manufacturer)
        
    def order_scans(self):
        order = np.argsort(self._instance_nums)
        self.scans = np.array(self.scans)
        self.scans = self.scans[order, :, :]
        
    def anim_scans(self):
        fig, ax = plt.subplots()
        for i, scan in enumerate(self.scans):
            img = self.concat_scan_seg(i)
            plt.imshow(img, cmap=plt.cm.bone)
            clear_output(wait=True)
            display(fig)
        plt.show()
        
    def show_scans(self):        
        # setup plot layout
        n_scans = self.scans.shape[0]
        cols = int(np.ceil(np.power(n_scans, 1/3)))
        rows = cols * 2
        if cols * rows < n_scans:
            rows += 1
        fig, ax = plt.subplots(rows, cols, figsize=[12,12])
        
        for i in range(cols * rows):
            row_ind = int(i / cols)
            col_ind = int(i % cols)
            if i < n_scans:
                img = self.concat_scan_seg(i)
                ax[row_ind, col_ind].set_title('slice %d' % (i + 1))
                ax[row_ind, col_ind].imshow(img, cmap=plt.cm.bone)
            ax[row_ind, col_ind].axis('off')
        plt.show()
        
    def concat_scan_seg(self, i):
        scan = self.scans[i]
        # we have two classes, so we can multiply by the half of the max
        seg = self.seg[:, :, i] * (scan.max()/2)
        return np.hstack([scan, seg])