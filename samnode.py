import sys
sys.path.append('./SAM')
sys.path.append('./')


from argparse import Namespace
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from .SAM.datasets.augmentations import AgeTransformer
from .SAM.utils.common import tensor2im
from .SAM.models.psp import pSp

class SamNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "image_in" : ("IMAGE", {}) },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    CATEGORY = "SAM"
    FUNCTION = "age"

    @staticmethod
    def run_alignment(image_path):
        import dlib
        from SAM.scripts.align_all_parallel import align_face
        predictor = dlib.shape_predictor("../models/SAM/shape_predictor_68_face_landmarks.dat")
        aligned_image = align_face(filepath=image_path, predictor=predictor)
        print("Aligned image has shape: {}".format(aligned_image.size))
        return aligned_image
    
    @staticmethod
    def run_on_batch(inputs, net):
        result_batch = net(inputs.to("cuda").float(), randomize_noise=False, resize=False)
        return result_batch

    def age(self, image_in):
        
        EXPERIMENT_ARGS = {            
            "model_path": "../models/SAM/sam_ffhq_aging.pt",            
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
        original_image = Image.open(image_in).convert("RGB")
        original_image.resize((256, 256))
        aligned_image = SamNode.run_alignment(image_in)
        aligned_image.resize((256, 256))
        img_transforms = EXPERIMENT_ARGS['transform']
        input_image = img_transforms(aligned_image)
        target_age = 70
        age_transformer = AgeTransformer(target_age)
        with torch.no_grad():
            input_image_age = [age_transformer(input_image.cpu()).to('cuda')]
            input_image_age = torch.stack(input_image_age)
            result_tensor = SamNode.run_on_batch(input_image_age, net)[0]
            result_image = tensor2im(result_tensor)
            result_image = Image.fromarray(result_image)

        return (result_image,)


if __name__ == "__main__":
    sn = SamNode()
    image_in = sys.argv[1]
    result = sn.age(image_in)
    pprint.pprint(result)
    result[0].show()
    result[0].save("result.jpg")
