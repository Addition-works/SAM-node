import sys
import os
import uuid
import folder_paths
import torch
import torchvision.transforms as transforms
from argparse import Namespace
from PIL import Image

# Add SAM to the Python path
sampath = os.path.join(folder_paths.base_path, 'custom_nodes/SAM-node/SAM')
nodepath = os.path.join(folder_paths.base_path, 'custom_nodes/SAM-node')
sys.path.append(os.path.abspath(sampath))
sys.path.append(os.path.abspath(nodepath))

from SAM.models.psp import pSp
from SAM.datasets.augmentations import AgeTransformer
from SAM.utils.common import tensor2im

class SamNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_in": ("IMAGE", {}),
                "target_age": ("INT", {"default": 70, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("aged_image", "aligned_face")
    CATEGORY = "SAM"
    FUNCTION = "age"

    @staticmethod
    def run_alignment(image_path):
        import dlib
        from SAM.scripts.align_all_parallel import align_face
        model_path = os.path.join(folder_paths.models_dir, "SAM/shape_predictor_68_face_landmarks.dat")
        predictor = dlib.shape_predictor(model_path)
        aligned_image = align_face(filepath=image_path, predictor=predictor)
        return aligned_image
    
    @staticmethod
    def run_on_batch(inputs, net):
        return net(inputs.to("cuda").float(), randomize_noise=True, resize=False)

    def age(self, image_in, target_age):
        model_path = os.path.join(folder_paths.models_dir, "SAM/sam_ffhq_aging.pt")
        EXPERIMENT_ARGS = {            
            "model_path": model_path,            
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
        
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts = Namespace(**opts)
        net = pSp(opts)
        net.eval()
        net.cuda()
        
        image_in_permuted = image_in.permute(0, 3, 1, 2)
        image_in_squeezed = image_in_permuted.squeeze(0)
        original_image = transforms.ToPILImage()(image_in_squeezed)

        imgid = str(uuid.uuid4())
        original_image.save(f'{imgid}.jpg')
        aligned_image = self.run_alignment(f'{imgid}.jpg')
        aligned_image = aligned_image.resize((256, 256))
        
        img_transforms = EXPERIMENT_ARGS['transform']
        input_image = img_transforms(aligned_image)
        
        age_transformer = AgeTransformer(target_age=target_age)
        
        with torch.no_grad():
            input_image_age = age_transformer(input_image.cpu()).to('cuda').unsqueeze(0)
            result_tensor = self.run_on_batch(input_image_age, net)[0]
            result_image = tensor2im(result_tensor)
            result_tensor = transforms.ToTensor()(result_image)
            result_tensor = result_tensor.unsqueeze(0).permute(0, 2, 3, 1)
        
        # Convert aligned_image to tensor in the format expected by ComfyUI
        aligned_tensor = transforms.ToTensor()(aligned_image).unsqueeze(0).permute(0, 2, 3, 1)
        
        os.remove(f'{imgid}.jpg')
        return (result_tensor, aligned_tensor)