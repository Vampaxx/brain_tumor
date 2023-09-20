import os
import pandas as pd
import tensorflow as tf
from src.brain_tumor.config.configuration import ConfugarationManager

class TensorConversionImgMask:

    def __init__(self):
        self.config_    = ConfugarationManager()
        self.path       = self.config_.get_data_ingestion_config()

    def path_creation_for_files(self, data_type:str):
        self.data_type  = data_type
        image_path      = self.path.image_path
        mask_path       = self.path.mask_path

        self.image_files    = []
        self.mask_files     = []

        data_file_path = None

        if data_type    == 'train':
            data_file_path = self.path.train_data_path
        elif data_type  == 'test':
            data_file_path = self.path.test_data_path
        elif data_type  == 'val':
            data_file_path = self.path.val_data_path
        else:
            raise ValueError("Invalid data_type. Use 'train', 'test', or 'val'.")

        data = pd.read_csv(data_file_path)

        for _, row in data.iterrows():
            image_file  = os.path.join(image_path, f"{row[0]}")
            mask_file   = os.path.join(mask_path, f"{row[1]}")

            self.image_files.append(image_file)
            self.mask_files.append(mask_file)

        return self.image_files, self.mask_files
        
    def convert_img_mask_tf(self, data_type):
        image, mask = self.path_creation_for_files(data_type)
        data        = tf.data.Dataset.from_tensor_slices((image, mask))
        return data
    

if __name__ == '__main__':
    obj     = TensorConversionImgMask()
    data    = obj.convert_img_mask_tf("train")



    

    