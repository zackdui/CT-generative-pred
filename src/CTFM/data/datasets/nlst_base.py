# Portions of this file are adapted from:
# <Original Repo Name> (MIT License)
# Copyright (c) 2021 Peter Mikhael & Jeremy Wohlwend
#
# Modifications (c) 2025 Zack Duitz

import os
import traceback, warnings
import pickle, json
import numpy as np
from tqdm import tqdm
from collections import Counter
import copy
import torch
from torch.utils import data
from CTFM.data.utils import (
    METAFILE_NOTFOUND_ERR,
    LOAD_FAIL_MSG,
    DEVICE_ID,
    get_sample_loader,
)
# from CTFM.utils.registry import register_object
import pydicom
import torchio as tio

warnings.filterwarnings("ignore")

GOOGLE_SPLITS_FILENAME = "/data/rbg/shared/projects/sybil/google_data_splits.p"

CORRUPTED_PATHS = "/data/rbg/shared/datasets/NLST/NLST/corrupted_img_paths.pkl"

CT_ITEM_KEYS = [
    "pid",
    "exam",
    "series",
    "y_seq",
    "y_mask",
    "time_at_event",
    "cancer_laterality",
    "has_annotation",
    "origin_dataset",
    "device",
    "slice_thickness",
    "days_at_event",
    "pixel_spacing",
]

DEVICE_TO_NAME = {
    1: "GE MEDICAL SYSTEMS",
    2: "Philips",
    3: "SIEMENS",
    4: "TOSHIBA",
    "GE MEDICAL SYSTEMS": "GE MEDICAL SYSTEMS",
    "Philips": "Philips",
    "SIEMENS": "SIEMENS",
    "TOSHIBA": "TOSHIBA",
}


PID2DICOM_DIRECTORY = pickle.load(
    open("/data/rbg/shared/datasets/NLST/NLST/all_nlst_dicoms_pid2directory.p", "rb")
)

def crop_or_pad_hw_only(target_w: int, target_h: int, padding_mode=0):
    return tio.Lambda(
        lambda s: tio.CropOrPad(
            (target_w, target_h, s.shape[3]),  # preserve depth
            padding_mode=padding_mode,
        )(s)
    )

# @register_object("nlst", "dataset")
class NLST_Survival_Dataset(data.Dataset):
    def __init__(self, args, split_group):
        """
        NLST Dataset
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        super(NLST_Survival_Dataset, self).__init__()

        self.split_group = split_group
        self.args = args
        self._num_images = args.num_images  # number of slices in each volume
        self._max_followup = args.max_followup

        try:
            self.metadata_json = json.load(open(args.dataset_file_path, "r"))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.dataset_file_path, e))

        self.input_loader = get_sample_loader(split_group, args)
        self.always_resample_pixel_spacing = (args.resample_pixel_spacing) and (
            split_group in ["dev", "test"]
        )
        if args.resample_pixel_spacing:
            img_size = args.img_size
            self.resample_transform = tio.transforms.Resample(
                target=tuple(args.ct_pixel_spacing)
            )
            self.padding_transform_original = tio.transforms.CropOrPad(
                target_shape=tuple(img_size + [args.num_images]), padding_mode=0
            ) # replaces args.num_images with none so the number of images could vary

            self.padding_transform = crop_or_pad_hw_only(
                target_w=img_size[0],
                target_h=img_size[1],
                padding_mode=0,
            )

            self.fake_padding_transform = tio.transforms.CropOrPad(
                target_shape=tuple([25, 25] + [1]), padding_mode=0
            ) # replaces args.num_images with none so the number of images could vary

        self.use_original_pad = getattr(args, "original_pad", False)

        if self.args.region_annotations_filepath:
            self.annotations_metadata = json.load(
                open(self.args.region_annotations_filepath, "r")
            )
        else:
            self.annotations_metadata = {}

        self.dataset = self.create_dataset(split_group)
        if len(self.dataset) == 0:
            return

        print(self.get_summary_statement(self.dataset, split_group))

        if args.class_bal:
            label_dist = [d[args.class_bal_key] for d in self.dataset]
            label_counts = Counter(label_dist)
            weight_per_label = 1.0 / len(label_counts)
            label_weights = {
                label: weight_per_label / count for label, count in label_counts.items()
            }

            print("Class counts are: {}".format(label_counts))
            print("Label weights are {}".format(label_weights))
            self.weights = [label_weights[d[args.class_bal_key]] for d in self.dataset]
        self.metadata_json = None

    def create_dataset(self, split_group):
        """
        Gets the dataset from the paths and labels in the json.
        Arguments:
            split_group(str): One of ['train'|'dev'|'test'].
        Returns:
            The dataset as a dictionary with img paths, label,
            and additional information regarding exam or participant
        """
       
        self.corrupted_paths = self.CORRUPTED_PATHS["paths"]
        self.corrupted_series = self.CORRUPTED_PATHS["series"]
        
        if self.args.assign_splits:
            np.random.seed(self.args.cross_val_seed)
            self.assign_splits(self.metadata_json)

        dataset = []

        for mrn_row in tqdm(self.metadata_json, position=0, ncols=100):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row["split"],
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            if not split == split_group:
                continue

            for exam_dict in exams:

                for series_id, series_dict in exam_dict["image_series"].items():
                    if self.skip_sample(series_id, series_dict, exam_dict, pt_metadata):
                        continue

                    sample = self.get_volume_dict(
                        series_id, series_dict, exam_dict, pt_metadata, pid, split
                    )
                    if len(sample) == 0:
                        continue

                    dataset.append(sample)

        return dataset

    def get_thinnest_cut(self, exam_dict):
        # volume that is not thin cut might be the one annotated; or there are multiple volumes with same num slices, so:
        # use annotated if available, otherwise use thinnest cut
        possibly_annotated_series = [
            s in self.annotations_metadata
            for s in list(exam_dict["image_series"].keys())
        ]
        series_lengths = [
            len(exam_dict["image_series"][series_id]["paths"])
            for series_id in exam_dict["image_series"].keys()
        ]
        thinnest_series_len = max(series_lengths)
        thinnest_series_id = [
            k
            for k, v in exam_dict["image_series"].items()
            if len(v["paths"]) == thinnest_series_len
        ]
        if any(possibly_annotated_series):
            thinnest_series_id = list(exam_dict["image_series"].keys())[
                possibly_annotated_series.index(1)
            ]
        else:
            thinnest_series_id = thinnest_series_id[0]
        return thinnest_series_id

    def skip_sample(self, series_id, series_dict, exam_dict, pt_metadata):
        series_data = series_dict["series_data"]
        if "reconthickness" not in series_data:
            slice_thickness = series_dict["slice_thickness"]
            screen_timepoint = exam_dict["screen_timepoint"]
            is_localizer = False
        elif len(series_data):
            # check if screen is localizer screen or not enough images
            is_localizer = self.is_localizer(series_data)

            # check if restricting to specific slice thicknesses
            slice_thickness = series_data["reconthickness"][0]
            screen_timepoint = series_data["study_yr"][0]

        if slice_thickness is None:
            # print("Here and slice_thickness is None")
            return True
        if self.args.slice_thickness_filter is None:
            print("slice thickness argument is None")
        wrong_thickness = (self.args.slice_thickness_filter is not None) and (
            slice_thickness > self.args.slice_thickness_filter or (slice_thickness < 0)
        )

        # check if valid label (info is not missing)

        bad_label = not self.check_label(pt_metadata, screen_timepoint)

        # invalid label
        if not bad_label:
            y, _, _, time_at_event, _ = self.get_label(pt_metadata, screen_timepoint)
            invalid_label = (y == -1) or (time_at_event < 0)
        else:
            invalid_label = False

        insufficient_slices = len(series_dict["paths"]) < self.args.min_num_images

        if (
            is_localizer
            or wrong_thickness
            or bad_label
            or invalid_label
            or insufficient_slices
        ):
            return True
        else:
            return False

    def get_volume_dict(
        self, series_id, series_dict, exam_dict, pt_metadata, pid, split
    ):
        img_paths = series_dict["paths"]
        slice_locations = series_dict["img_position"]
        series_data = series_dict["series_data"]
        # In the test split the following line commented out line had a typo and therefore it was replaced
        # device = DEVICE_ID[DEVICE_TO_NAME[series_data["manufacturer"][0]]]
        manufacturer = series_data["manufacturer"][0]
        device_name = DEVICE_TO_NAME.get(manufacturer, "UNKNOWN")
        if device_name == "UNKNOWN":
            print(f"Unknown device manufacturer: {manufacturer}")
        device = DEVICE_ID.get(device_name, -1)
        screen_timepoint = series_data["study_yr"][0]
        assert screen_timepoint == exam_dict["screen_timepoint"]

        if series_id in self.corrupted_series:
            if any([path in self.corrupted_paths for path in img_paths]):
                uncorrupted_imgs = np.where(
                    [path not in self.corrupted_paths for path in img_paths]
                )[0]
                img_paths = np.array(img_paths)[uncorrupted_imgs].tolist()
                slice_locations = np.array(slice_locations)[uncorrupted_imgs].tolist()

        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations
        )

        if not sorted_img_paths[0].startswith(self.args.img_dir):
            sorted_img_paths = [
                self.args.img_dir
                + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                for path in sorted_img_paths
            ]

        if sorted_img_paths[0].endswith(".dcm.png"):
            sorted_img_paths = [p.replace(".dcm.png", ".png") for p in sorted_img_paths]

        if (
            self.args.img_file_type == "dicom"
        ):  # ! NOTE: removing file extension affects get_ct_annotations mapping path to annotation
            sorted_img_paths = [
                (
                    PID2DICOM_DIRECTORY[pid]
                    + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
                )
                .replace(".png", ".dcm")
                .replace(
                    "/Mounts/rbg-storage1/datasets", "/data/rbg/shared/datasets/NLST"
                )
                for path in sorted_img_paths
            ]

        y, y_seq, y_mask, time_at_event, days_at_event = self.get_label(pt_metadata, screen_timepoint)

        exam_int = int(
            "{}{}{}".format(
                pid, int(screen_timepoint), int(series_id.split(".")[-1][-3:])
            )
        )
        sample = {
            "paths": sorted_img_paths,
            "slice_locations": sorted_slice_locs,
            "y": int(y),
            "time_at_event": time_at_event,
            "days_at_event": days_at_event,
            "y_seq": y_seq,
            "y_mask": y_mask,
            "exam_str": "{}_{}".format(exam_dict["exam"], series_id),
            "exam": exam_int,
            "accession": exam_dict["accession_number"],
            "series": series_id,
            "screen_timepoint": screen_timepoint,
            "pid": pid,
            "device": device,
            "num_original_slices": len(series_dict["paths"]),
            "pixel_spacing": series_dict["pixel_spacing"]
            + [series_dict["slice_thickness"]],
            "slice_thickness": self.get_slice_thickness_class(
                series_dict["slice_thickness"]
            ),
        }

        return sample

    def check_label(self, pt_metadata, screen_timepoint):
        valid_days_since_rand = (
            pt_metadata["scr_days{}".format(screen_timepoint)][0] > -1
        )
        valid_days_to_cancer = pt_metadata["candx_days"][0] > -1
        valid_followup = pt_metadata["fup_days"][0] > -1
        return (valid_days_since_rand) and (valid_days_to_cancer or valid_followup)

    def get_label(self, pt_metadata, screen_timepoint):
        days_since_rand = pt_metadata["scr_days{}".format(screen_timepoint)][0]
        days_to_cancer_since_rand = pt_metadata["candx_days"][0]
        days_to_cancer = days_to_cancer_since_rand - days_since_rand
        years_to_cancer = (
            int(days_to_cancer // 365) if days_to_cancer_since_rand > -1 else 100
        )
        days_to_last_followup = int(pt_metadata["fup_days"][0] - days_since_rand)
        years_to_last_followup = days_to_last_followup // 365
        y = years_to_cancer < self.args.max_followup
        y_seq = np.zeros(self.args.max_followup)
        cancer_timepoint = pt_metadata["cancyr"][0]
        if y:
            if years_to_cancer > -1:
                assert screen_timepoint <= cancer_timepoint
            time_at_event = years_to_cancer
            y_seq[years_to_cancer:] = 1
            days_at_event = days_to_cancer
        else:
            time_at_event = min(years_to_last_followup, self.args.max_followup - 1)
            days_at_event = min(days_to_last_followup, (self.args.max_followup - 1) * 365)
            days_at_event = days_to_last_followup
        y_mask = np.array(
            [1] * (time_at_event + 1)
            + [0] * (self.args.max_followup - (time_at_event + 1))
        )
        assert len(y_mask) == self.args.max_followup
        return y, y_seq.astype("float64"), y_mask.astype("float64"), time_at_event, days_at_event

    def is_localizer(self, series_dict):
        is_localizer = (
            (series_dict["imageclass"][0] == 0)
            or ("LOCALIZER" in series_dict["imagetype"][0])
            or ("TOP" in series_dict["imagetype"][0])
        )
        return is_localizer

    def get_pixel_spacing(self, dcm_path):
        """Get slice thickness and row/col spacing

        Args:
            path (str): path to sample png file in the series

        Returns:
            pixel spacing: [thickness, spacing[0], spacing[1]]
                thickness (float): CT slice thickness
                spacing (list): spacing along x and y axes
        """
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        spacing = [float(d) for d in dcm.PixelSpacing] + [float(dcm.SliceThickness)]
        return spacing

    def order_slices(self, img_paths, slice_locations, reverse=False):
        sorted_ids = np.argsort(slice_locations)
        if reverse:
            sorted_ids = sorted_ids[::-1]
        sorted_img_paths = np.array(img_paths)[sorted_ids].tolist()
        sorted_slice_locs = np.sort(slice_locations).tolist()

        return sorted_img_paths, sorted_slice_locs

    def assign_splits(self, meta):
        if self.args.split_type == "institution_split":
            self.assign_institutions_splits(meta)
        elif self.args.split_type == "random":
            for idx in range(len(meta)):
                meta[idx]["split"] = np.random.choice(
                    ["train", "dev", "test"], p=self.args.split_probs
                )
    @property
    def CORRUPTED_PATHS(self):
        return pickle.load(open(CORRUPTED_PATHS, "rb"))

    def get_summary_statement(self, dataset, split_group):
        summary = "Contructed NLST CT Cancer Risk {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
        class_balance = Counter([d["y"] for d in dataset])
        exams = set([d["exam"] for d in dataset])
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(
            split_group, len(dataset), len(exams), len(patients), class_balance
        )
        statement += "\n" + "Censor Times: {}".format(
            Counter([d["time_at_event"] for d in dataset])
        )
        statement
        return statement

    @property
    def GOOGLE_SPLITS(self):
        return pickle.load(open(GOOGLE_SPLITS_FILENAME, "rb"))

    def get_ct_annotations(self, sample):
        # correct empty lists of annotations
        if sample["series"] in self.annotations_metadata:
            self.annotations_metadata[sample["series"]] = {
                k: v
                for k, v in self.annotations_metadata[sample["series"]].items()
                if len(v) > 0
            }

        if sample["series"] in self.annotations_metadata:
            # store annotation(s) data (x,y,width,height) for each slice
            if (
                self.args.img_file_type == "dicom"
            ):  # no file extension, so os.path.splitext breaks behavior
                sample["annotations"] = [
                    {
                        "image_annotations": self.annotations_metadata[
                            sample["series"]
                        ].get(os.path.basename(path), None)
                    }
                    for path in sample["paths"]
                ]
            else:  # expects file extension to exist, so use os.path.splitext
                sample["annotations"] = [
                    {
                        "image_annotations": self.annotations_metadata[
                            sample["series"]
                        ].get(os.path.splitext(os.path.basename(path))[0], None)
                    }
                    for path in sample["paths"]
                ]
        else:
            sample["annotations"] = [
                {"image_annotations": None} for path in sample["paths"]
            ]
        return sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = copy.deepcopy(self.dataset[index])
        # print("Here in get item")
        # print(type(sample))
        return self.process_item(sample)

    def process_item(self, sample):
        if self.args.use_annotations:
            sample = self.get_ct_annotations(sample)
        try:
            item = {}
            input_dict = self.get_images(sample["paths"], sample)

            x = input_dict["input"]

            if self.args.use_annotations:
                mask = torch.abs(input_dict["mask"])
                mask_area = mask.sum(dim=(-1, -2))
                item["volume_annotations"] = mask_area[0] / max(1, mask_area.sum())
                item["annotation_areas"] = mask_area[0] / (
                    mask.shape[-2] * mask.shape[-1]
                )
                mask_area = mask_area.unsqueeze(-1).unsqueeze(-1)
                mask_area[mask_area == 0] = 1
                item["image_annotations"] = mask / mask_area
                item["has_annotation"] = item["volume_annotations"].sum() > 0

            if self.args.use_risk_factors:
                item["risk_factors"] = sample["risk_factors"]

            item["x"] = x
            item["y"] = sample["y"]
            # item["day_at_evant"]
            for key in CT_ITEM_KEYS:
                if key in sample:
                    item[key] = sample[key]

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

    def get_images(self, paths, sample):
        """
        Returns a stack of transformed images by their absolute paths.
        If cache is used - transformed images will be loaded if available,
        and saved to cache if not.
        """
        out_dict = {}
        if self.args.fix_seed_for_multi_image_augmentations:
            sample["seed"] = np.random.randint(0, 2**32 - 1)

        # get images for multi image input
        s = copy.deepcopy(sample)
        input_dicts = []
        for e, path in enumerate(paths):
            if self.args.use_annotations:
                s["annotations"] = sample["annotations"][e]
            input_dicts.append(self.input_loader.get_image(path, s))

        images = [i["input"] for i in input_dicts]
        input_arr = self.reshape_images(images)
        if self.args.use_annotations:
            masks = [i["mask"] for i in input_dicts]
            mask_arr = self.reshape_images(masks) if self.args.use_annotations else None

        # resample pixel spacing
        resample_now = (
            self.args.resample_pixel_spacing_prob > np.random.uniform()
        ) and self.args.resample_pixel_spacing
        self.always_resample_pixel_spacing = True
        if self.always_resample_pixel_spacing or resample_now:
            spacing = torch.tensor(sample["pixel_spacing"] + [1])
            input_arr = tio.ScalarImage(
                affine=torch.diag(spacing),
                tensor=input_arr.permute(0, 2, 3, 1),
            )
            input_arr = self.resample_transform(input_arr)
            if input_arr.data.shape[3] <= self.args.min_num_images:
                input_arr = self.fake_padding_transform(input_arr.data)
            elif self.use_original_pad:
                input_arr = self.padding_transform_original(input_arr.data)
            else:
                input_arr = self.padding_transform(input_arr.data)

            if self.args.use_annotations:
                mask_arr = tio.ScalarImage(
                    affine=torch.diag(spacing),
                    tensor=mask_arr.permute(0, 2, 3, 1),
                )
                mask_arr = self.resample_transform(mask_arr)
                # mask_arr = self.padding_transform(mask_arr.data)

        if self.args.resample_pixel_spacing:
            out_dict["input"] = input_arr.data.permute(0, 3, 1, 2)
            if self.args.use_annotations:
                out_dict["mask"] = mask_arr.data.permute(0, 3, 1, 2)
        else:
            out_dict["input"] = input_arr
            if self.args.use_annotations:
                out_dict["mask"] = mask_arr

        return out_dict

    def reshape_images(self, images):
        images = [im.unsqueeze(0).unsqueeze(0) for im in images]
        images = torch.cat(images, dim=0)

        # Convert from (T, C, H, W) to (C, T, H, W)
        images = images.permute(1, 0, 2, 3)
        return images

    def get_slice_thickness_class(self, thickness):
        BINS = [1, 1.5, 2, 2.5]
        for i, tau in enumerate(BINS):
            if thickness <= tau:
                return i
        if self.args.slice_thickness_filter is not None:
            raise ValueError("THICKNESS > 2.5")
        return 4




