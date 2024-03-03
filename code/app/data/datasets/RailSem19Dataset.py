from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
import cv2

# Paper: https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Zendel_RailSem19_A_Dataset_for_Semantic_Rail_Scene_Understanding_CVPRW_2019_paper.pdf
# Data: https://wilddash.cc/railsem19

class RailSem19Dataset(Dataset):

    def __init__(self, root_path):
        self.root_path = root_path
        self.images_paths, self.masks_paths, self.objects_paths = self.__get_files_paths(root_path)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.root_path + 'jpgs/rs19_val/' + self.images_paths[idx], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(self.root_path + 'uint8/rs19_val/' + self.masks_paths[idx], cv2.IMREAD_UNCHANGED)
        return image, mask

    def __get_files_paths(self, root_path):

        images_folder = root_path + 'jpgs/rs19_val/'
        masks_folder = root_path + 'uint8/rs19_val/'
        objects_folder = root_path + 'jsons/rs19_val/'

        images_paths = [f for f in listdir(images_folder) if isfile(join(images_folder, f))]
        masks_paths = [f for f in listdir(masks_folder) if isfile(join(masks_folder, f))]
        objects_paths = [f for f in listdir(objects_folder) if isfile(join(objects_folder, f))]

        return images_paths, masks_paths, objects_paths
    

