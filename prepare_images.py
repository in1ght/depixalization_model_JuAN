import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision
from to_grayscale import to_grayscale
from pixelization import pixelization
import pickle
import os


from pathlib import Path


class Prepare_Images():
    def __init__(self,image_dir: str,dtype = np.uint8):
        self.raw_images = [os.path.abspath(file_) for file_ in Path(image_dir).rglob('*') if file_.is_file() and str(file_).endswith(".jpg")]
        self.prepared = []
        self.way = None
        self.dtype = dtype
    

    def __getitem__(self,index: int) -> tuple:
        list_of_transforms = [transforms.Resize(size=(64,64),interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                              transforms.CenterCrop(size=(64,64))]
        
        transform_chain = transforms.Compose(list_of_transforms)
        
        image_raw = Image.open(self.raw_images[index])

        target_image = transform_chain(image_raw)

        target_image = np.array(target_image, dtype = self.dtype)

        target_image_ = to_grayscale(target_image)
        
        mean, std = target_image_.mean(), target_image_.std()

        pixelated_arr, known_array = pixelization(target_image_,index)
        
        return target_image_, pixelated_arr, known_array, mean, std



    def prepare_images(self) -> list:
        target_image = []
        pixelated_arr = []
        known_array = []
        mean_std = []
        for i in range(len(self.raw_images)):
            target, pixelated, known, mean, std = self.__getitem__(i)
            target_image.append(target)
            pixelated_arr.append(pixelated)
            known_array.append(known)
            mean_std.append((mean,std))

        self.prepared = target_image

        return target_image, pixelated_arr, known_array, mean_std



    def write_prepared_images(self, way):
        images_tuples = self.prepare_images_to_list(self.raw_images)

        with open('list_of_images.pickle', 'wb') as handle:
            pickle.dump(images_tuples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.way = way

        return True


    def augment_data(self):
        pass


    def read_prepared_images(self):
        if self.way == None:
            return False
        
        with open(f'{self.way}list_of_images.pickle', 'rb') as handle:
            unserialized_data = pickle.load(handle)

        return unserialized_data

    def __len__(self):
        return len(self.prepared)