import torch
from torch.utils.data import Dataset, DataLoader

import cv2, os
from pathlib import Path
from typing import Tuple
import random
import pandas as pd

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)
random.seed(seed)

class DataPrefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)  # Tạo iterator cho DataLoader
        self.stream = torch.cuda.Stream()  # Tạo stream CUDA để tải trước dữ liệu
        self.preload()  # Tải trước dữ liệu đầu tiên

    def preload(self):
        try:
            self.next_data = next(self.loader)  # Lấy batch tiếp theo từ DataLoader
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            # Tải dữ liệu vào GPU
            self.next_input = self.next_data[0].cuda(non_blocking=True)
            self.next_target = self.next_data[1].cuda(non_blocking=True)

    def next(self):
        # Đồng bộ hóa stream hiện tại với stream prefetch
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()  # Tải trước batch tiếp theo
        return input, target


class CustomExrDataset(Dataset):
    
    def __init__(self, dataset_dir:str, transform, type='normalmap', train = True):
        '''
            type = ['normalmap', 'depthmap', 'albedo']
        '''
        self.metadata_path = os.path.join(dataset_dir, 'train_set.csv') if train else os.path.join(dataset_dir, 'test_set.csv')
        split = 'train' if train else 'test'
        if type == 'normalmap':
            dataset_dir = Path(dataset_dir, 'Normal_Map', split)
        elif type == 'depthmap':
            dataset_dir = Path(dataset_dir, 'Depth_Map', split)
        elif type == 'albedo':
            dataset_dir = Path(dataset_dir, 'Albedo', split)
        else:
            raise Exception(f'Sai trường type: {type}')
        
        self.paths = list(Path(dataset_dir).glob("*/*.exr"))
        self.transform = transform
        self.type = type
        self.classes = sorted(os.listdir(dataset_dir))
        
        self.weightclass = self.__calculate_weight_class()
        
    def __len__(self):
        return len(self.paths)
    
    # Nhận vào index mà dataloader muốn lấy
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        image_path = self.paths[index]
        numpy_image = self.__load_numpy_image(image_path)
        gender, spectacles, facial_hair, pose, occlusion, emotion = self.__extract_csv(image_path)
        label = image_path.parent.name
        label_index = self.classes.index(label)
        
        if self.transform:
            numpy_image = self.transform(image = numpy_image)['image']
        
        X = torch.from_numpy(numpy_image).permute(2,0,1)
        y = torch.tensor([label_index, gender, spectacles, facial_hair, pose, occlusion, emotion], dtype=torch.int)
        return X, y
        
    def __load_numpy_image(self, image_path:str):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        if self.type in ['albedo', 'depthmap']:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image

    def __extract_csv(self, image_path):
        id = image_path.parent.name
        session = image_path.stem
        df = pd.read_csv(self.metadata_path)
        
        # Lọc dữ liệu theo ID và session
        filtered_data = df[(df['id'] == int(id)) & (df['session'] == session)]
        
        # Kiểm tra nếu không có hoặc có nhiều hơn 1 dòng được trả về
        if filtered_data.shape[0] != 1:
            raise Exception(f"Tìm thấy {filtered_data.shape[0]} row có {id} và {session} trong file {self.metadata_path}")
        
        row = filtered_data.iloc[0]  # Lấy dòng đầu tiên (vì chỉ có 1 dòng được trả về)
        
        return row["Gender"], row["Spectacles"], row["Facial_Hair"], row["Pose"], row["Occlusion"], row["Emotion"]

    def __calculate_weight_class(self):
        df = pd.read_csv(self.metadata_path)
        
        label_columns = ['id', 'Gender', 'Spectacles', 'Facial_Hair', 'Pose', 'Occlusion', 'Emotion']
        
         # Calculate the relative frequency and alpha_t for each class in each label column
        weight_class = {}
        for column in label_columns:
            value_counts = df[column].value_counts(normalize=True)  # Tần suất tương đối
            alpha_t = (1 / value_counts) / (1 / value_counts).sum()  # Normalize để tổng bằng 1
            weight_class[column] = alpha_t.to_dict()
        
        return weight_class

        
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


def create_multitask_datafetcher(config, train_transform, test_transform) -> Tuple[DataLoader, DataLoader]:
    
    train_data = CustomExrDataset(config['dataset_dir'], train_transform, config['type'])
    test_data = CustomExrDataset(config['dataset_dir'], test_transform, config['type'], train=False)

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
    
    # train_datafetcher = DataPrefetcher(train_dataloader)
    # test_datafetcher = DataPrefetcher(test_dataloader)
    
    
    return train_dataloader, test_dataloader, train_data.weightclass


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
