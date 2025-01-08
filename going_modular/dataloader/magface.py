import torch
from torch.utils.data import Dataset, DataLoader, Sampler

import cv2, os
from pathlib import Path
from typing import Tuple
import random

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)
random.seed(seed)

class CustomExrDataset(Dataset):
    
    def __init__(self, dataset_dir:str, transform, type='normalmap'):
        '''
            type = ['normalmap', 'depthmap', 'albedo']
        '''
        self.paths = list(Path(dataset_dir).glob("*/*.exr"))
        self.transform = transform
        self.type = type
        self.classes = sorted(os.listdir(dataset_dir))
        
    def __len__(self):
        return len(self.paths)
    
    # Nhận vào index mà dataloader muốn lấy
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        numpy_image = self.__load_numpy_image(index)
        label = self.paths[index].parent.name
        label_index = self.classes.index(label)
        
        if self.transform:
            numpy_image = self.transform(image = numpy_image)['image']
            
        return torch.from_numpy(numpy_image).permute(2,0,1), label_index
        
    def __load_numpy_image(self, index:int):
        image = cv2.imread(self.paths[index], cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"Failed to load image at {self.paths[index]}")
        if self.type in ['albedo', 'depthmap']:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image

    
class MultiModalExrDataset(Dataset):
    def __init__(self, dataset_dir:str, transform=None, is_train=True):
        split = 'train' if is_train else 'test'
        self.albedo_dir = Path(dataset_dir) / 'Albedo' / split
        self.depth_dir = Path(dataset_dir) / 'Depth_Map' / split
        self.normal_dir = Path(dataset_dir) / 'Normal_Map' / split
        
        self.transform = transform
        self.classes = sorted(os.listdir(self.albedo_dir))
        
        # Collect paths for each modality
        self.data = []
        for class_name in self.classes:
            albedo_class_dir = self.albedo_dir / class_name
            depth_class_dir = self.depth_dir / class_name
            normal_class_dir = self.normal_dir / class_name

            albedo_files = sorted(list(albedo_class_dir.glob("*.exr")))
            depth_files = sorted(list(depth_class_dir.glob("*.exr")))
            normal_files = sorted(list(normal_class_dir.glob("*.exr")))

            assert len(albedo_files) == len(normal_files) == len(depth_files), (
                f"Mismatch in number of files for class {class_name}: Albedo({len(albedo_files)}), "
                f"Normal({len(normal_files)}), Depth({len(depth_files)})"
            )
            class_index = self.classes.index(class_name)
            for albedo_path, normal_path, depth_path in zip(albedo_files, normal_files, depth_files):
                self.data.append((albedo_path, normal_path, depth_path, class_index))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        albedo_path, normal_path, depth_path, class_index = self.data[index]
        
        albedo = self.__load_numpy_image(albedo_path)
        normal = self.__load_numpy_image(normal_path)
        depth = self.__load_numpy_image(depth_path)
        
        if self.transform:
            transformed = self.transform(image=albedo, depthmap=depth, normalmap=normal)
            albedo = transformed['image']
            depth = transformed['depthmap']
            normal = transformed['normalmap']
        
        # Stack các tensor lại thành một tensor duy nhất
        X = torch.stack((
            torch.from_numpy(albedo).permute(2, 0, 1), 
            torch.from_numpy(depth).permute(2, 0, 1),
            torch.from_numpy(normal).permute(2, 0, 1)
        ), dim=0)
        
        return X, class_index
        
    def __load_numpy_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image


def create_magface_dataloader(config, train_transform, test_transform) -> Tuple[DataLoader, DataLoader]:
    
    train_data = CustomExrDataset(config['train_dir'], train_transform, config['type'])
    test_data = CustomExrDataset(config['test_dir'], test_transform, config['type'])

    train_dataloader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    
    return train_dataloader, test_dataloader


def create_concat_magface_dataloader(config, train_transform, test_transform) ->Tuple[DataLoader, DataLoader]:
    train_data = MultiModalExrDataset(config['dataset_dir'], train_transform)
    test_data = MultiModalExrDataset(config['dataset_dir'], test_transform, is_train=False)
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        # sampler=sampler,
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    return train_dataloader, test_dataloader
