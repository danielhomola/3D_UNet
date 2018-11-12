"""
Classes for loading and visualising MRI scans with their corresponding
segmentation file for all patients in a dataset.
"""

import re
import pickle

import skimage
import pydicom
import nrrd
import numpy as np
import tensorflow as tf
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
            dict: a dict of :class:`src.data_utils.Patient` objects.
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
        for patient in patients.values():
            patient.order_scans()
        
        return patients

    def preprocess_dataset(self, width=128, height=128, depths=(24, 32)):
        """
        Scales each scan in the dataset to be between zero and one, then
        resizes all scans and targets to have the same width and height.

        It also turns the target into a one-hot encoded tensor of shape:
        [depth, width, height, num_classes].

        Finally, it ensures that all patients have the same number of depth(s),
        i.e. their tensors have the same  dimensions. Defaults: 24 and 32.
        This is to ensure that a 3D U-Net with depth 4 can be built, i.e. the
        downsampled layer of the maxpooling and the upsampled layer of the
        transponsed convolution are guaranteed to have the same depth.

        Args:
            width (int): Width to resize all scans and targets in dataset.
            height (int): Height to resize all scans and targets in dataset.
            depths (tuple <int>): Tuple of acceptable depths.
        """
        for patient in self.patients.values():
            patient.normalise_scans()
            patient.resize_and_reshape(width=width, height=height)
            patient.adjust_depth(depths=depths)
            patient.preprocessed = True

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
            dict: a dict of :class:`src.data_utils.Patient` objects.
        """
        return pickle.load(open(path, 'rb'))

    def create_tf_dataset(self):
        """
        Creates a TensorFlow DataSet from the DataSet. Note, this has to be
        run after all the MRI scans have been rescaled to the same size. They
        can have different depths, i.e. number of scans however.

        Returns:
            :class:`tf.data.Dataset`: TensorFlow dataset.
        """

        # extract all scans and segmentation images from every patient
        x_all = [p.scans for p in self.patients.values()]
        y_all = [p.seg for p in self.patients.values()]

        # extract depths, width and height, num_classes from the dataset
        depths = list(set([x.shape[0] for x in x_all]))
        _, width, height = x_all[0].shape
        num_classes = y_all[0].shape[-1]

        datasets = []
        for depth in depths:
            # select patients with the right number of scans
            x_depth = [x for x in x_all if x.shape[0] == depth]
            y_depth = [y for y in y_all if y.shape[0] == depth]
            # reshape dataset from list of 3d volumes into 5d volume
            x_depth = np.concatenate(x_depth).reshape(
                (-1, depth, width, height, 1)
            )
            y_depth = np.concatenate(y_depth).reshape(
                (-1, depth, width, height, num_classes)
            )
            datasets.append(
                tf.data.Dataset.from_tensor_slices((x_depth, y_depth))
            )

        # concatenate all tf datasets into a single one
        if len(datasets) == 1:
            merged_dataset = datasets[0]
        else:
            merged_dataset = datasets.pop(0)
            for dataset in datasets:
                merged_dataset = merged_dataset.concatenate(dataset)
        return merged_dataset


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
        # make depth the first dim as with the scans
        self.seg = np.moveaxis(seg, -1, 0)
        self._instance_nums = list()
        self.thicknesses = set()
        self.manufacturers = set()
        self.add_scan(scan)
        self.preprocessed = False
        
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

    def normalise_scans(self):
        """
        Scales each scan in the dataset to be between zero and one.
        """
        scans = self.scans.astype(np.float32)

        for i, scan in enumerate(scans):
            scan_min = np.min(scan)
            scan_max = np.max(scan)

            # avoid dividing by zero
            if scan_max != scan_min:
                scans[i] = (scan - scan_min) / (scan_max - scan_min)
            else:
                scans[i] = scan * 0
        self.scans = scans

    def resize_and_reshape(self, width, height):
        """
        Resizes each scan and target segmentation image of a patient to a
        given width and height. It also turns the target into a one-hot
        encoded tensor of shape: [depth, width, height, num_classes].

        Args:
            width (int): Width to resize all scans and targets in dataset.
            height (int): Height to resize all scans and targets in dataset.
        """
        # resize scans
        depth = self.scans.shape[0]
        scans = skimage.transform.resize(
            image=self.scans,
            output_shape=(depth, width, height)
        )
        self.scans = scans

        # turns target into one-hot tensor, adopted from:
        # https://stackoverflow.com/a/36960495
        n_classes = self.seg.max() + 1
        seg = (np.arange(n_classes) == self.seg[..., None]).astype(bool)

        # resize targets while preserving their boolean nature
        seg = skimage.img_as_bool(
            skimage.transform.resize(
                image=seg,
                output_shape=(depth, width, height, n_classes)
            )
        )
        self.seg = seg.astype(int)

    def adjust_depth(self, depths):
        """
        There's a wide range of scan numbers across the patients. We need to
        unify these so they can be fed into the network.

        This is to ensure that a 3D U-Net with depth 4 can be built, i.e. the
        downsampled layer of the maxpooling and the upsampled layer of the
        transponsed convolution are guaranteed to have the same depth.

        Note this function can only be run once the target tensor has been
        converted to a one hot encoded version with `resize_and_reshape`.

        Args:
            depths (tuple <int>): Tuple of acceptable depths.
        """
        depth, width, height = self.scans.shape

        # find which acceptable depth we should use
        depths = np.array(depths)
        new_depth = depths[np.argmin(np.abs(depths - depth))]
        depth_delta = np.abs(new_depth - depth)

        # we need to adjust the depth
        if depth_delta != 0:
            # pad scans and seg with repeating last scan
            if new_depth > depth:
                to_add = np.repeat(self.scans[-1:, :, :], depth_delta, axis=0)
                self.scans = np.concatenate((self.scans, to_add), axis=0)

                to_add = np.repeat(self.seg[-1:, :, :, :], depth_delta, axis=0)
                self.seg = np.concatenate((self.seg, to_add), axis=0)
            # delete last few scans and segs
            else:
                self.scans = self.scans[:new_depth, :, :]
                self.seg = self.seg[:new_depth, :, :, :]

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

        If we have preprocessed data, i.e. the target is a one hot encoded
        tensor, we use the 2nd class for visualisation.

        Args:
            i (int): Index of scan in the patient's list of scans.

        """
        scan = self.scans[i]
        # if we have preprocessed data use class 2 from target/segmentation
        if self.preprocessed:
            seg = self.seg[i, :, :, 1] * scan.max()
        else:
            seg = self.seg[i] * (scan.max() / 2)
        return np.hstack([scan, seg])

