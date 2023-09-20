from src.brain_tumor.utils.common import read_yaml,create_directories
from src.brain_tumor.constants import *
from src.brain_tumor.entity.config_entity import DataIngestionConfig




class ConfugarationManager:

    def __init__(self,
                 config_file_path=CONFIG_FILE_PATH,
                 params_file_path=PARAMS_FILE_PATH):
        
        # Assuming read_yaml and create_directories are custom functions in your module
        self.config_ = read_yaml(config_file_path)
        self.params_ = read_yaml(params_file_path)
        create_directories([self.config_.artifacts_root]) # by use of Configbox, can all attributes directly

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config_ = self.config_.data_ingestion 
        create_directories([config_.root_dir])

        data_ingestion_config = DataIngestionConfig(
            train_data_path = Path(config_.train_path),
            test_data_path  = Path(config_.test_path), 
            val_data_path   = Path(config_.val_path),
            raw_data_path   = Path(config_.csv_file_path),
            image_path      = Path(config_.image_path),
            mask_path       = Path(config_.mask_path))
        
        return data_ingestion_config
