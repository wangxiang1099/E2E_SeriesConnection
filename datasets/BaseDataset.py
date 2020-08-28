
from torch.utils.data import Dataset

# you need to prepare dataset as this  if muti >=3  you need to realize transform s3 ....
class BaseDataset(Dataset):
    
    def __init__(self):
        self.image_path = None
        self.load_label()

    def __len__(self):
        return len(self.image_path)
    
    def load_target(self):
        raise NotImplementedError

    def load_image(self):
        raise NotImplementedError
    
    def transform_image(self):
        raise NotImplementedError

    def transform_s1_target(self):
        raise NotImplementedError

    def transform_s2_target(self):
        raise NotImplementedError
    
    def __getitem__(self, idx, vis=False):
        
        image_path = self.image_path[idx]
        image_dir, image_name = os.path.split(os.path.splitext(image_path)[0])
        target = self.read_label(os.path.join(image_dir, image_name + '.txt'))

        mode = os.path.basename(image_dir)
        name = mode + '_' + image_name

        image = cv2.imread(image_path)

        image = self.transform_image(image)
        s1_target = self.transform_s1_target()
        s2_target = self.transform_s2_target()

        return name, image, s1_target, s2_target






