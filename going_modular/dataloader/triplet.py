# Code anh lâm
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
import torch
from pathlib import Path
from typing import Tuple
from torchvision import transforms
from torchvision.transforms import InterpolationMode

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


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

    
class TripletDataset(Dataset):
    
    def __init__(self, data_dir, transform=None, type = 'albedo', train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train

        self.image_paths = []
        self.labels = []

        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for image_name in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(int(label))

        self.type = type
        self.labels = np.array(self.labels)
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

        if not self.train:
            random_state = np.random.RandomState(29)

            self.test_triplets = [[i,
                                   random_state.choice(self.label_to_indices[self.labels[i]]),
                                   random_state.choice(self.label_to_indices[
                                       np.random.choice(list(self.labels_set - set([self.labels[i]])))
                                   ])
                                  ]
                                 for i in range(len(self.image_paths))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.train:
            img1_path = self.image_paths[index]
            label1 = self.labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2_path = self.image_paths[positive_index]
            img3_path = self.image_paths[negative_index]
        else:
            img1_path = self.image_paths[self.test_triplets[index][0]]
            img2_path = self.image_paths[self.test_triplets[index][1]]
            img3_path = self.image_paths[self.test_triplets[index][2]]

        if self.type == 'normalmap':
            cvtColor = cv2.COLOR_BGR2RGB
        else:
            cvtColor = cv2.COLOR_GRAY2BGR
            
        img1 = cv2.cvtColor(cv2.imread(img1_path, cv2.IMREAD_UNCHANGED), cvtColor)
        img2 = cv2.cvtColor(cv2.imread(img2_path, cv2.IMREAD_UNCHANGED), cvtColor)
        img3 = cv2.cvtColor(cv2.imread(img3_path, cv2.IMREAD_UNCHANGED), cvtColor)

        if self.transform is not None:
            img1 = self.transform(image=img1)['image']
            img2 = self.transform(image=img2)['image']
            img3 = self.transform(image=img3)['image']

        # Stack các tensor lại thành một tensor duy nhất
        X = torch.stack((
            torch.from_numpy(img1).permute(2,0,1), 
            torch.from_numpy(img2).permute(2,0,1), 
            torch.from_numpy(img3).permute(2,0,1)
        ), dim=0)
        
        # image 1 và 2 cùng class, 3 là negative
        return X


# Concat Lam
class CustomExrDatasetConcat(Dataset):
    
    def __init__(self, data_dir: str, transform, train=True):
        split = 'train' if train else 'test'
        self.normal_dir = Path(data_dir, 'Normal_Map', split)
        self.albedo_dir = Path(data_dir, 'Albedo', split)
        self.normal_paths = sorted(list(self.normal_dir.glob("*/*.exr")))
        self.albedo_paths = sorted(list(self.albedo_dir.glob("*/*.exr")))
        
        if len(self.normal_paths) != len(self.albedo_paths):
            raise ValueError("Mismatch in number of files between Normal_Map and Albedo directories.")

        self.transform = transform
        self.classes = sorted(os.listdir(self.normal_dir))
        
    def __len__(self):
        return len(self.normal_paths)
    
    # Nhận vào index mà dataloader muốn lấy
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        numpy_normalmap, numpy_albedo = self.__load_numpy_image(index)
        label = self.normal_paths[index].parent.name
        label_index = self.classes.index(label)
        
        if self.transform:
            numpy_normalmap = self.transform(image = numpy_normalmap)['image']
            numpy_albedo = self.transform(image = numpy_albedo)['image']
        
        X = torch.stack((
            torch.from_numpy(numpy_normalmap).permute(2,0,1), 
            torch.from_numpy(numpy_albedo).permute(2,0,1), 
        ), dim=0)
        
        return X, label_index  

    def __load_numpy_image(self, index:int):
        normalmap = cv2.imread(self.normal_paths[index], cv2.IMREAD_UNCHANGED)
        albedo = cv2.imread(self.albedo_paths[index], cv2.IMREAD_UNCHANGED)
        
        if normalmap is None:
            raise ValueError(f"Failed to load image at {self.normal_paths[index]}")
        
        normalmap = cv2.cvtColor(normalmap, cv2.COLOR_BGR2RGB)
        albedo = cv2.cvtColor(albedo, cv2.COLOR_GRAY2RGB)
        
        return normalmap, albedo


class TripletDatasetConcat(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        split = 'train' if train else 'test'
        self.normalmap_dir = os.path.join(data_dir, 'Normal_Map', split)
        self.albedo_dir = os.path.join(data_dir, 'Albedo', split)
        self.transform = transform
        self.train = train

        self.image_paths = []
        self.labels = []

        for label in os.listdir(self.normalmap_dir):
            normalmap_id_path = os.path.join(self.normalmap_dir, label)
            albedo_id_path = os.path.join(self.albedo_dir, label)
            for image_name in os.listdir(normalmap_id_path):
                normalmap_path = os.path.join(normalmap_id_path, image_name)
                albedo_path = os.path.join(albedo_id_path, image_name)
                self.image_paths.append((normalmap_path, albedo_path))
                self.labels.append(int(label))
                    
        self.labels = np.array(self.labels)
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        if not self.train:
            random_state = np.random.RandomState(29)

            self.test_triplets = [[i,
                                   random_state.choice(self.label_to_indices[self.labels[i]]),
                                   random_state.choice(self.label_to_indices[
                                       np.random.choice(list(self.labels_set - set([self.labels[i]])))
                                   ])
                                  ]
                                 for i in range(len(self.image_paths))]
            
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.train:
            img1_path = self.image_paths[index]
            label1 = self.labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2_path = self.image_paths[positive_index]
            img3_path = self.image_paths[negative_index]
        else:
            img1_path = self.image_paths[self.test_triplets[index][0]]
            img2_path = self.image_paths[self.test_triplets[index][1]]
            img3_path = self.image_paths[self.test_triplets[index][2]]

        img1 = cv2.cvtColor(cv2.imread(img1_path[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(img1_path[1], cv2.IMREAD_UNCHANGED), cv2.COLOR_GRAY2BGR)
        img3 = cv2.cvtColor(cv2.imread(img2_path[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img4 = cv2.cvtColor(cv2.imread(img2_path[1], cv2.IMREAD_UNCHANGED), cv2.COLOR_GRAY2BGR)
        img5 = cv2.cvtColor(cv2.imread(img3_path[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img6 = cv2.cvtColor(cv2.imread(img3_path[1], cv2.IMREAD_UNCHANGED), cv2.COLOR_GRAY2BGR)

        if self.transform is not None:
            img1 = self.transform(image=img1)['image']
            img2 = self.transform(image=img2)['image']
            img3 = self.transform(image=img3)['image']
            img4 = self.transform(image=img4)['image']
            img5 = self.transform(image=img5)['image']
            img6 = self.transform(image=img6)['image']
        
        X = torch.stack((
            torch.from_numpy(img1).permute(2,0,1), 
            torch.from_numpy(img2).permute(2,0,1), 
            torch.from_numpy(img3).permute(2,0,1),
            torch.from_numpy(img4).permute(2,0,1), 
            torch.from_numpy(img5).permute(2,0,1), 
            torch.from_numpy(img6).permute(2,0,1)
        ), dim=0)
        
        return X


# Concat 2 type of data
class CustomExrDatasetConCatV2(Dataset):
    
    def __init__(self, data_dir_1: str, type_1: str, data_dir_2: str, type_2: str, transform, train=True):
        split = 'train' if train else 'test'
        self.data_dir_1 = Path(data_dir_1, split)
        self.data_dir_2 = Path(data_dir_2, split)
        self.type_1_paths = sorted(list(self.data_dir_1.glob("*/*.exr")))
        self.type_2_paths = sorted(list(self.data_dir_2.glob("*/*.exr")))
        
        if len(self.type_1_paths) != len(self.type_2_paths):
            raise ValueError(f"Mismatch in number of files between {data_dir_1} and {data_dir_2} directories.")

        self.type_1 = type_1
        self.type_2 = type_2
        self.transform = transform
        self.classes = sorted(os.listdir(self.data_dir_1))
        
    def __len__(self):
        return len(self.type_1_paths)
    
    # Nhận vào index mà dataloader muốn lấy
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        numpy_image_1, numpy_image_2 = self.__load_numpy_images(index)
        label = self.type_1_paths[index].parent.name
        label_index = self.classes.index(label)
        
        if self.transform is not None:
            transformed = self.transform(image=numpy_image_1, image_1=numpy_image_2)
            numpy_image_1 = transformed['image']
            numpy_image_2 = transformed['image_1']
            
        X = torch.stack((
            torch.from_numpy(numpy_image_1).permute(2,0,1), 
            torch.from_numpy(numpy_image_2).permute(2,0,1), 
        ), dim=0)
        
        return X, label_index  
        
    def __load_numpy_images(self, index:int):
        image_1 = cv2.imread(self.type_1_paths[index], cv2.IMREAD_UNCHANGED)
        image_2 = cv2.imread(self.type_2_paths[index], cv2.IMREAD_UNCHANGED)
        
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_GRAY2RGB) if self.type_1 in ['albedo, depthmap'] else cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_GRAY2RGB) if self.type_2 in ['albedo, depthmap'] else cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
        
        return image_1, image_2


class TripletDatasetConcatV2(Dataset):
    def __init__(self, data_dir_1: str, type_1: str, data_dir_2: str, type_2: str, transform=None, train=True):
        split = 'train' if train else 'test'
        self.data_dir_1 = os.path.join(data_dir_1, split)
        self.data_dir_2 = os.path.join(data_dir_2, split)
        self.transform = transform
        self.train = train
        self.type_1 = type_1
        self.type_2 = type_2

        self.image_paths = []
        self.labels = []

        for label in os.listdir(self.data_dir_1):
            type_1_id_path = os.path.join(self.data_dir_1, label)
            type_2_id_path = os.path.join(self.data_dir_2, label)
            for image_name in os.listdir(type_1_id_path):
                type_1_path = os.path.join(type_1_id_path, image_name)
                type_2_path = os.path.join(type_2_id_path, image_name)
                self.image_paths.append((type_1_path, type_2_path))
                self.labels.append(int(label))
                    
        self.labels = np.array(self.labels)
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        if not self.train:
            random_state = np.random.RandomState(29)

            self.test_triplets = [[i,
                                   random_state.choice(self.label_to_indices[self.labels[i]]),
                                   random_state.choice(self.label_to_indices[
                                       np.random.choice(list(self.labels_set - set([self.labels[i]])))
                                   ])
                                  ]
                                 for i in range(len(self.image_paths))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.train:
            img1_path = self.image_paths[index]
            label1 = self.labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2_path = self.image_paths[positive_index]
            img3_path = self.image_paths[negative_index]
        else:
            img1_path = self.image_paths[self.test_triplets[index][0]]
            img2_path = self.image_paths[self.test_triplets[index][1]]
            img3_path = self.image_paths[self.test_triplets[index][2]]

        img1 = cv2.cvtColor(cv2.imread(img1_path[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_GRAY2RGB) if self.type_1 in ['albedo', 'depthmap'] else cv2.cvtColor(cv2.imread(img1_path[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(img1_path[1], cv2.IMREAD_UNCHANGED), cv2.COLOR_GRAY2RGB) if self.type_2 in ['albedo', 'depthmap'] else cv2.cvtColor(cv2.imread(img1_path[1], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img3 = cv2.cvtColor(cv2.imread(img2_path[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_GRAY2RGB) if self.type_1 in ['albedo', 'depthmap'] else cv2.cvtColor(cv2.imread(img2_path[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img4 = cv2.cvtColor(cv2.imread(img2_path[1], cv2.IMREAD_UNCHANGED), cv2.COLOR_GRAY2RGB) if self.type_2 in ['albedo', 'depthmap'] else cv2.cvtColor(cv2.imread(img2_path[1], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img5 = cv2.cvtColor(cv2.imread(img3_path[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_GRAY2RGB) if self.type_1 in ['albedo', 'depthmap'] else cv2.cvtColor(cv2.imread(img3_path[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img6 = cv2.cvtColor(cv2.imread(img3_path[1], cv2.IMREAD_UNCHANGED), cv2.COLOR_GRAY2RGB) if self.type_2 in ['albedo', 'depthmap'] else cv2.cvtColor(cv2.imread(img3_path[1], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=img1, image_1=img2)
            img1 = transformed['image']
            img2 = transformed['image_1']
            transformed = self.transform(image=img3, image_1=img4)            
            img3 = transformed['image']
            img4 = transformed['image_1']
            transformed = self.transform(image=img5, image_1=img6)
            img5 = transformed['image']
            img6 = transformed['image_1']
        
        X = torch.stack((
            torch.from_numpy(img1).permute(2,0,1), 
            torch.from_numpy(img2).permute(2,0,1), 
            torch.from_numpy(img3).permute(2,0,1),
            torch.from_numpy(img4).permute(2,0,1), 
            torch.from_numpy(img5).permute(2,0,1), 
            torch.from_numpy(img6).permute(2,0,1),
        ), dim=0)
        
        return X
    
# Concat 3 type of data
class CustomExrDatasetConCatV3(Dataset):
    
    def __init__(self, data_dir: str, transform, train=True):
        split = 'train' if train else 'test'
        self.normal_dir = Path(data_dir, 'Normal_Map', split)
        self.albedo_dir = Path(data_dir, 'Albedo', split)
        self.depth_dir = Path(data_dir, 'Albedo', split)
        self.normal_paths = sorted(list(self.normal_dir.glob("*/*.exr")))
        self.albedo_paths = sorted(list(self.albedo_dir.glob("*/*.exr")))
        self.depth_paths = sorted(list(self.albedo_dir.glob("*/*.exr")))
        
        if len(self.normal_paths) != len(self.albedo_paths) != len(self.depth_dir):
            raise ValueError("Mismatch in number of files between Normal_Map and Albedo directories.")

        self.transform = transform
        self.classes = sorted(os.listdir(self.normal_dir))
        
    def __len__(self):
        return len(self.normal_paths)
    
    # Nhận vào index mà dataloader muốn lấy
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        numpy_normalmap, numpy_albedo, numpy_depthmap = self.__load_numpy_image(index)
        label = self.normal_paths[index].parent.name
        label_index = self.classes.index(label)
        
        if self.transform is not None:
            transformed = self.transform(image=numpy_normalmap, albedo=numpy_albedo, depthmap=numpy_depthmap)
            numpy_normalmap = transformed['image']
            numpy_albedo = transformed['albedo']
            numpy_depthmap = transformed['depthmap']
            
        X = torch.stack((
            torch.from_numpy(numpy_normalmap).permute(2,0,1), 
            torch.from_numpy(numpy_albedo).permute(2,0,1), 
            torch.from_numpy(numpy_depthmap).permute(2,0,1), 
        ), dim=0)
        
        return X, label_index  
        
    def __load_numpy_image(self, index:int):
        normalmap = cv2.imread(self.normal_paths[index], cv2.IMREAD_UNCHANGED)
        albedo = cv2.imread(self.albedo_paths[index], cv2.IMREAD_UNCHANGED)
        depthmap = cv2.imread(self.depth_paths[index], cv2.IMREAD_UNCHANGED)
        
        normalmap = cv2.cvtColor(normalmap, cv2.COLOR_BGR2RGB)
        albedo = cv2.cvtColor(albedo, cv2.COLOR_GRAY2RGB)
        depthmap = cv2.cvtColor(depthmap, cv2.COLOR_GRAY2RGB)
        
        return normalmap, albedo, depthmap


class TripletDatasetConcatV3(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        split = 'train' if train else 'test'
        self.normalmap_dir = os.path.join(data_dir, 'Normal_Map', split)
        self.albedo_dir = os.path.join(data_dir, 'Albedo', split)
        self.depthmap_dir = os.path.join(data_dir, 'Depth_Map', split)
        self.transform = transform
        self.train = train

        self.image_paths = []
        self.labels = []

        for label in os.listdir(self.normalmap_dir):
            normalmap_id_path = os.path.join(self.normalmap_dir, label)
            albedo_id_path = os.path.join(self.albedo_dir, label)
            depthmap_id_path = os.path.join(self.depthmap_dir, label)
            for image_name in os.listdir(normalmap_id_path):
                normalmap_path = os.path.join(normalmap_id_path, image_name)
                albedo_path = os.path.join(albedo_id_path, image_name)
                depthmap_path = os.path.join(depthmap_id_path, image_name)
                self.image_paths.append((normalmap_path, albedo_path, depthmap_path))
                self.labels.append(int(label))
                    
        self.labels = np.array(self.labels)
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        if not self.train:
            random_state = np.random.RandomState(29)

            self.test_triplets = [[i,
                                   random_state.choice(self.label_to_indices[self.labels[i]]),
                                   random_state.choice(self.label_to_indices[
                                       np.random.choice(list(self.labels_set - set([self.labels[i]])))
                                   ])
                                  ]
                                 for i in range(len(self.image_paths))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.train:
            img1_path = self.image_paths[index]
            label1 = self.labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2_path = self.image_paths[positive_index]
            img3_path = self.image_paths[negative_index]
        else:
            img1_path = self.image_paths[self.test_triplets[index][0]]
            img2_path = self.image_paths[self.test_triplets[index][1]]
            img3_path = self.image_paths[self.test_triplets[index][2]]

        img1 = cv2.cvtColor(cv2.imread(img1_path[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(img1_path[1], cv2.IMREAD_UNCHANGED), cv2.COLOR_GRAY2BGR)
        img3 = cv2.cvtColor(cv2.imread(img1_path[2], cv2.IMREAD_UNCHANGED), cv2.COLOR_GRAY2BGR)
        img4 = cv2.cvtColor(cv2.imread(img2_path[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img5 = cv2.cvtColor(cv2.imread(img2_path[1], cv2.IMREAD_UNCHANGED), cv2.COLOR_GRAY2BGR)
        img6 = cv2.cvtColor(cv2.imread(img2_path[2], cv2.IMREAD_UNCHANGED), cv2.COLOR_GRAY2BGR)
        img7 = cv2.cvtColor(cv2.imread(img3_path[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img8 = cv2.cvtColor(cv2.imread(img3_path[1], cv2.IMREAD_UNCHANGED), cv2.COLOR_GRAY2BGR)
        img9 = cv2.cvtColor(cv2.imread(img3_path[2], cv2.IMREAD_UNCHANGED), cv2.COLOR_GRAY2BGR)

        if self.transform is not None:
            transformed = self.transform(image=img1, albedo=img2, depthmap=img3)
            img1 = transformed['image']
            img2 = transformed['albedo']
            img3 = transformed['depthmap']
            transformed = self.transform(image=img4, albedo=img5, depthmap=img6)
            img4 = transformed['image']
            img5 = transformed['albedo']
            img6 = transformed['depthmap']
            transformed = self.transform(image=img7, albedo=img8, depthmap=img9)
            img7 = transformed['image']
            img8 = transformed['albedo']
            img9 = transformed['depthmap']
        
        X = torch.stack((
            torch.from_numpy(img1).permute(2,0,1), 
            torch.from_numpy(img2).permute(2,0,1), 
            torch.from_numpy(img3).permute(2,0,1),
            torch.from_numpy(img4).permute(2,0,1), 
            torch.from_numpy(img5).permute(2,0,1), 
            torch.from_numpy(img6).permute(2,0,1),
            torch.from_numpy(img7).permute(2,0,1),
            torch.from_numpy(img8).permute(2,0,1),
            torch.from_numpy(img9).permute(2,0,1),
        ), dim=0)
        
        return X