# Your imports
import sys
import os
import uuid
import folder_paths
# Add SAM to the Python path
print('folder_paths.base_dir: ', folder_paths.base_path)
sampath = os.path.join(folder_paths.base_path, 'custom_nodes/SAM-node/SAM')
nodepath = os.path.join(folder_paths.base_path, 'custom_nodes/SAM-node')
print('sampath: ', sampath)
print('nodepath: ', nodepath)
sys.path.append(os.path.abspath(sampath))  # Path to SAM
sys.path.append(os.path.abspath(nodepath))     # Path to custom node root

# Optional: print sys.path to verify
print(sys.path)
from SAM.models.psp import pSp
from SAM.datasets.augmentations import AgeTransformer
from SAM.utils.common import tensor2im

from argparse import Namespace
import os
import sys
import pprint
import numpy as np
import torch
import torchvision.transforms as transforms

print("Imported SAM node")
print("Folder paths models: ", folder_paths.models_dir)


class SamNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "image_in" : ("IMAGE", {}) },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("aged_image", "face_image")
    CATEGORY = "SAM"
    FUNCTION = "age"

    @staticmethod
    def run_alignment(image_tensor):
        import dlib
        from SAM.scripts.align_all_parallel import align_face_tensor
        model_path = os.path.join(folder_paths.models_dir, "SAM/shape_predictor_68_face_landmarks.dat")
        print("Alignment model path: ", model_path)
        predictor = dlib.shape_predictor(model_path)
        aligned_image = align_face_tensor(image_tensor, predictor=predictor)  # Assuming align_face_tensor exists for tensor input
        print("Aligned image has shape: {}".format(aligned_image.shape))
        return aligned_image
    
    @staticmethod
    def run_on_batch(inputs, net):
        result_batch = net(inputs.to("cuda").float(), randomize_noise=False, resize=False)
        return result_batch

    def age(self, image_in):
        model_path = os.path.join(folder_paths.models_dir, "SAM/sam_ffhq_aging.pt")
        print("Aging model path: ", model_path)
        EXPERIMENT_ARGS = {            
            "model_path": model_path,            
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
        model_path = EXPERIMENT_ARGS['model_path']
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts = Namespace(**opts)
        net = pSp(opts)
        net.eval()
        net.cuda()
        print('SAM model successfully loaded!')
        print('image_in dimensions: ', image_in.size())
        image_in_permuted = image_in.permute(0, 3, 1, 2)  # Change from [B, H, W, C] to [B, C, H, W]
        image_in_squeezed = image_in_permuted.squeeze(0)  # Now squeeze the batch size

        # Use torchvision's transforms to resize the tensor
        transform_resize = transforms.Resize((256, 256))
        resized_image = transform_resize(image_in_squeezed)

        aligned_image = SamNode.run_alignment(resized_image)
        aligned_resized_image = transform_resize(aligned_image)  # Resize aligned image

        img_transforms = EXPERIMENT_ARGS['transform']
        input_image = img_transforms(aligned_resized_image)
        target_ages = [70]
        age_transformers = [AgeTransformer(target_age=age) for age in target_ages]        
        for age_transformer in age_transformers:
            print(f"Running on target age: {age_transformer.target_age}")
            with torch.no_grad():
                input_image_age = [age_transformer(input_image.cpu()).to('cuda')]
                input_image_age = torch.stack(input_image_age)
                result_tensor = SamNode.run_on_batch(input_image_age, net)[0]
                result_image = tensor2im(result_tensor)
                
                # convert that numpy image to tensor (skipping PIL)
                result_tensor = transforms.ToTensor()(result_image)
                result_tensor = result_tensor.unsqueeze(0)
                print(result_tensor.size())
                result_tensor = result_tensor.permute(0, 2, 3, 1)
                print(result_tensor.size())
                
        return (result_tensor,aligned_resized_image,)


if __name__ == "__main__":
    sn = SamNode()
    image_in = sys.argv[1]
    result = sn.age(image_in)
    pprint.pprint(result)
