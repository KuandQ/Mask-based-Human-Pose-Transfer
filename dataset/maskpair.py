import csv
import os

from .pair import PairBoneDataset, wrap_dict_name


class MaskPairBoneDataset(PairBoneDataset):
    @staticmethod
    def load_mask2pair_list(mask2pair_list_path):
        assert os.path.isfile(mask2pair_list_path)
        with open(mask2pair_list_path, "r") as f:
            f_csv = csv.reader(f)
            next(f_csv)
            mask2pair_list = [tuple(item) for item in f_csv]
            return mask2pair_list

    def __init__(self, mask2pair_list_path, *nargs, **kwargs):
        super().__init__(*nargs, **kwargs)
        self.mask2pair_list_path = mask2pair_list_path
        self.mask2pairs = self.load_mask2pair_list(mask2pair_list_path)

    def __getitem__(self, input_idx):
        mask2_p1_name, mask2_p2_name = self.mask2pairs[input_idx]
        mask2pair = wrap_dict_name(self.prepare_item(mask2_p1_name), "condition_")
        mask2pair.update(wrap_dict_name(self.prepare_item(mask2_p2_name), "target_"))

        return mask2pair

    def __len__(self):
        return len(self.mask2pairs)

    def __repr__(self):
        return """
{}(
    size: {},
    flip_rate: {},
    image_folder: {},
    bone_folder: {},
    mask2_folder: {}
    pair_list_path: {}
    mask2pair_list_path: {}
    transform: {}
)

    """.format(self.__class__, len(self), self.flip_rate, self.image_folder,
               self.bone_folder, self.mask2_folder, self.pair_list_path,
               self.mask2pair_list_path, self.transform)
