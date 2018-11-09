"""
Classes for loading and visualising MRI scans with their corresponding
segmentation file for all patients in a dataset.
"""

import re
import pickle

import pydicom
import nrrd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


class Dataset(object):
    """
    Object to collect all patients within the train/test set.
    """
    def __init__(self, scan_files, seg_files):
        """
        Class initialiser.

        Args:
            scan_files (list <str>): List of paths to MRI scan files.
            seg_files (list <str>): List of paths to segmentation files.
        """
        self.scan_files = scan_files
        self.seg_files = seg_files
        self.patients = self.build_dataset()
        self.patient_ids = list(self.patients.keys())

    def build_dataset(self):
        """
        Adds all patients that can be found in the provided path folders.
        Reads in all the MRI scans (multiple per patient) and their
        corresponding segmentation files (one per patient). Finally, it orders
        the scans of each patient.


        Returns:
            dict: a dict of :class:`src.data.data_utils.Patient` objects.
        """

        patients = dict()
        
        # read in all scans as a list - one file for each scan
        scans = [pydicom.dcmread(scan) for scan in self.scan_files]
        
        # read in all the segmentation files - one file for each patient
        segs = {
            seg.split('/')[-1]: nrrd.read(seg)[0] for seg in self.seg_files}
        
        # build dict of patient objects
        for i, scan in enumerate(scans):
            if scan.PatientID not in patients:
                # unfortunately, the PatientID cannot be trusted as sometimes 
                # it doesn't correspond to the appropriate nddr file, so we use 
                # regex to extract patient_id from the folder of the DICOM file
                pat_folder_regex = re.compile(r'\/(Prostate[a-zA-Z0-9\-]+)\/')
                patient_folder = re.search(
                    pat_folder_regex, self.scan_files[i]).group(1)
                seg_file = segs['%s.nrrd' % patient_folder]
                patients[scan.PatientID] = Patient(scan=scan, seg=seg_file)
            else:
                patients[scan.PatientID].add_scan(scan=scan)
        
        # sort scans within each patient
        for patient in patients.keys():
            patients[patient].order_scans()
        
        return patients

    def save_dataset(self, path):
        """
        Saves the dataset as a pickled object.

        Args:
            path (str): Full path and object name used for saving the dataset.
        """
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load_dataset(path):
        """
        Loads the dataset from a pickled file.

        Args:
            path (str): Full path to the pickled dataset.

        Returns:
            dict: a dict of :class:`src.data.data_utils.Patient` objects.
        """
        return pickle.load(open(path, 'rb'))

        
class Patient(object):
    """
    Basic object to store all slices of a patient in the study.
    """
    def __init__(self, scan, seg):
        """
        Class initialiser.

        Args:
            scan (str): Path to one MRI scan file.
            seg (str): Path to the segmentation file.
        """
        self.scans = list()
        self.seg = seg
        self._instance_nums = list()
        self.thicknesses = set()
        self.manufacturers = set()
        self.add_scan(scan)
        
    def add_scan(self, scan):
        """
        Adds one more scan to a patient's list of scans. It also saves it's
        manufacturer, slice thickness and instance number, i.e. the index of
        the scan in the sequence.

        Args:
            scan (:class:`pydicom.dataset.FileDataset'): A loaded MRI scan.
        """

        self.scans.append(scan.pixel_array)
        self._instance_nums.append(int(scan.InstanceNumber))
        self.thicknesses.add(int(scan.SliceThickness))
        self.manufacturers.add(scan.Manufacturer)
        
    def order_scans(self):
        """
        Orders the scans of a patient according to the imaging sequence.
        """
        order = np.argsort(self._instance_nums)
        self.scans = np.array(self.scans)
        self.scans = self.scans[order, :, :]
        
    def anim_scans(self):
        """
        Generates an animation for IPython, visualising all the scans of a
        patient.
        """
        fig, ax = plt.subplots()
        for i, scan in enumerate(self.scans):
            img = self.concat_scan_seg(i)
            plt.imshow(img, cmap=plt.cm.bone)
            clear_output(wait=True)
            display(fig)
        plt.show()
        
    def show_scans(self):
        """
        Generates a tiled image, visualising all the scans of a patient.
        """
        n_scans = self.scans.shape[0]
        cols = int(np.ceil(np.power(n_scans, 1/3)))
        rows = cols * 2
        if cols * rows < n_scans:
            rows += 1
        fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
        
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
        """
        Helper function, that concatenates the MRI image with its corresponding
        segmentation image file and rescales the latter so their colours are
        comparable.
        Args:
            i (int): Index of scan in the patient's list of scans.

        """
        scan = self.scans[i]
        seg = self.seg[:, :, i] * (scan.max()/2)
        return np.hstack([scan, seg])