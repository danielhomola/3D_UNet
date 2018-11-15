"""
Classes for loading and visualising MRI scans with their corresponding
segmentation file for all patients in a dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import pickle
import logging

import skimage
import pydicom
import nrrd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

logger = logging.getLogger('tensorflow')


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

            logger.info('Reading in and parsing scan %d/%d' % (i, len(scans)))
        
        # sort scans within each patient
        for patient in patients.values():
            patient.order_scans()
        
        return patients

    def preprocess_dataset(self, resize=True, width=128, height=128,
                           max_scans=32):
        """
        Scales each scan in the dataset to be between zero and one, then
        resizes all scans and targets to have the same width and height.

        It also turns the target into a one-hot encoded tensor of shape:
        [depth, width, height, num_classes].

        Finally it ensures that all patients have at maximum `max_scans` scans.
        Patients with fewer scans will be padded with zeros. Extra scans of
        patients who have more (around 5% of the dataset) will be discarded.
        This is to ensure that a 3D U-Net with depth 4 can be built, i.e. the
        downsampled layer of the maxpooling and the upsampled layer of the
        transponsed convolution are guaranteed to have the same depth and
        shortcut connections can be made between them.

        Args:
            resize (bool): Whether to resize or not the scans.
            width (int): Width to resize all scans and targets in dataset.
            height (int): Height to resize all scans and targets in dataset.
            max_scans (tuple <int>): Maximum number of scans to keep.
        """
        for i, patient in enumerate(self.patients.values()):
            patient.normalise_scans()
            patient.resize_reshape(resize=resize, width=width, height=height)
            patient.adjust_depth(max_scans=max_scans)
            patient.preprocessed = True

            logger.info('Preprocessing data of patient %d/%d' % (
                i, len(self.patients.values())))

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

    def create_tf_dataset(self, resized=True, num_classes=3):
        """
        Creates a TensorFlow DataSet from the DataSet. Note, this has to
        be run after all the MRI scans have been rescaled to the same size.
        They can have different depths, i.e. number of scans however.

        Args:
            resized (bool): Whether dealing with a uniformly resized scans.
            num_classes (int): Number of classes in segmentation files.

        Returns:
            :class:`tf.data.Dataset`: TensorFlow dataset.
        """

        # extract all scans and segmentation images from every patient
        scans_segs = [(p.scans, p.seg) for p in self.patients.values()]
        if resized:
            _, width, height = scans_segs[0][0].shape
            x_shape = [None, width, height, 1]
            y_shape = [None, width, height, num_classes]
        else:
            x_shape = [None, None, None, 1]
            y_shape = [None, None, None, num_classes]

        def gen_scans_segs():
            """
            Generator function for dataset creation.
            """
            for s in scans_segs:
                # if each image is different sized, we need to get w, h here
                _, width, height = s[0].shape

                # add channel dimension to scans
                x = s[0].reshape((-1, width, height, 1))
                yield (x, s[1])

        return tf.data.Dataset.from_generator(
            generator=gen_scans_segs,
            output_types=(tf.float32, tf.int32),
            output_shapes=(x_shape, y_shape)
        )


class Patient(object):
    """
    Basic object to store all slices of a patient in the study.
    """
    def __init__(self, scan, seg):
        """
        Class initialiser.

        Args:
            scan (:class:`pydicom.dataset.FileDataset'): A loaded MRI scan.
            seg (:class:`numpy.ndarray`): A loaded segmentation file.
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

    def resize_reshape(self, resize, width, height):
        """
        Resizes each scan and target segmentation image of a patient to a
        given width and height. It also turns the target into a one-hot
        encoded tensor of shape: [depth, width, height, num_classes].

        Args:
            resize (bool): Whether to resize or not the scans.
            width (int): Width to resize all scans and targets in dataset.
            height (int): Height to resize all scans and targets in dataset.
        """
        # resize scans
        if resize:
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
        if resize:
            seg = skimage.img_as_bool(
                skimage.transform.resize(
                    image=seg,
                    output_shape=(depth, width, height, n_classes)
                )
            )
        self.seg = seg.astype(int)

    def adjust_depth(self, max_scans):
        """
        There's a wide range of scan numbers across the patients. We need to
        unify these so they can be fed into the network. We discard extra
        scans (i.e. more than the `max_scans`) and patients with less will be
        padded with zeros by TensorFlow.

        This is to ensure that a 3D U-Net with depth 4 can be built, i.e. the
        downsampled layer of the maxpooling and the upsampled layer of the
        transponsed convolution are guaranteed to have the same depth.

        Note this function can only be run once the target tensor has been
        converted to a one hot encoded version with `resize_and_reshape`.

        Args:
            max_scans (tuple <int>): Maximum number of scans to keep.
        """
        self.scans = self.scans[:max_scans]
        self.seg = self.seg[:max_scans]

    def patient_tile_scans(self):
        """
        Generates a tiled image, visualising all the scans of a patient.
        """
        Patient.tile_scans(self.scans, self.seg, self.preprocessed)

    def patient_anim_scans(self):
        """
        Generates an animation for IPython, visualising all the scans of a
        patient.
        """
        Patient.anim_scans(self.scans, self.seg, self.preprocessed)

    @staticmethod
    def anim_scans(scans, seg, preprocessed):
        """
        Generates an animation for IPython, visualising all the scans of a
        patient.

        Args:
            scans (:class:`numpy.array`): MRI image with shape
                [depth, width, height]
            seg (:class:`numpy.array`): MRI image segmentation with shape
                [depth, width, height]
            preprocessed (bool): Whether the scans been preprocessed.
        """
        fig, ax = plt.subplots()
        for i, scan in enumerate(scans):
            img = Patient.concat_scan_seg(scans, seg, i, preprocessed)
            plt.imshow(img, cmap=plt.cm.bone)
            clear_output(wait=True)
            display(fig)
        plt.show()

    @staticmethod
    def tile_scans(scans, seg, preprocessed):
        """
        Generates a tiled image, visualising all the scans of a patient.

        Args:
            scans (:class:`numpy.array`): MRI image with shape
                [depth, width, height]
            seg (:class:`numpy.array`): MRI image segmentation with shape
                [depth, width, height]
            preprocessed (bool): Whether the scans been preprocessed.
        """

        n_scans = scans.shape[0]
        cols = int(np.ceil(np.power(n_scans, 1/3)))
        rows = cols * 2
        if cols * rows < n_scans:
            rows += 1
        fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
        
        for i in range(cols * rows):
            row_ind = int(i / cols)
            col_ind = int(i % cols)
            if i < n_scans:
                img = Patient.concat_scan_seg(scans, seg, i, preprocessed)
                ax[row_ind, col_ind].set_title('slice %d' % (i + 1))
                ax[row_ind, col_ind].imshow(img, cmap=plt.cm.bone)
            ax[row_ind, col_ind].axis('off')
        plt.show()

    @staticmethod
    def concat_scan_seg(scans, seg, i, preprocessed):
        """
        Helper function, that concatenates the MRI image with its corresponding
        segmentation and rescales the latter so their colours are comparable.

        If we have preprocessed data, i.e. the target is a one hot encoded
        tensor, we use the 2nd class for visualisation.

        Args:
            scans (:class:`numpy.array`): MRI image with shape
                [depth, width, height]
            seg (:class:`numpy.array`): MRI image segmentation with shape
                [depth, width, height]
            i (int): Index of scan in the patient's list of scans.
            preprocessed (bool): Whether the scans been preprocessed.
        """
        scan = scans[i]
        # if we have preprocessed data use class 2 from target/segmentation
        if preprocessed:
            seg = seg[i, :, :, 1] * scan.max()
        else:
            seg = seg[i] * (scan.max() / 2)
        return np.hstack([scan, seg])

