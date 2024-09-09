# Your imports
import sys
import os
import uuid

# Add SAM to the Python path
sys.path.append(os.path.abspath('/workspace/ComfyUI/custom_nodes/SAM-node/SAM'))  # Path to SAM
sys.path.append(os.path.abspath('/workspace/ComfyUI/custom_nodes/SAM-node'))     # Path to custom node root


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
from PIL import Image
import torch
import torchvision.transforms as transforms
import folder_paths

print("Imported SAM node")
print("Folder paths models: ", folder_paths.models_dir)


def tensor2tensor(var):
    # Move to CPU and detach from computation graph
    var = var.cpu().detach()

    # Transpose the tensor from [C, H, W] to [H, W, C] if needed
    var = var.permute(1, 2, 0)

    # Scale the tensor from [-1, 1] to [0, 1]
    var = (var + 1) / 2

    # Clamp values to ensure they are within [0, 1] range
    var = torch.clamp(var, 0, 1)

    # Convert the values to the range [0, 255] for image-like output, if needed
    var = var * 255

    # Convert the tensor to an integer type
    var = var.type(torch.uint8)

    return var  # This is the processed tensor


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
        model_path = os.path.join(folder_paths.models_dir, "SAM/shape_predictor_68_face_landmarks.dat")
        print("Alignment model path: ", model_path)
        predictor = dlib.shape_predictor(model_path)
        aligned_image = align_face(filepath=image_path, predictor=predictor)
        print("Aligned image has shape: {}".format(aligned_image.size))
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
        
        image_in_permuted = image_in.permute(0, 3, 1, 2)  # Change from [B, H, W, C] to [B, C, H, W]
        image_in_squeezed = image_in_permuted.squeeze(0)  # Now squeeze the batch size
        original_image = transforms.ToPILImage()(image_in_squeezed)

        # save image

        imgid = str(uuid.uuid4())
        original_image.save(f'{imgid}.jpg')
        original_image.resize((256, 256))
        aligned_image = SamNode.run_alignment(f'{imgid}.jpg')
        aligned_image.resize((256, 256))
        img_transforms = EXPERIMENT_ARGS['transform']
        input_image = img_transforms(aligned_image)
        target_ages = [70]
        age_transformers = [AgeTransformer(target_age=age) for age in target_ages]        
        for age_transformer in age_transformers:
            print(f"Running on target age: {age_transformer.target_age}")
            with torch.no_grad():
                input_image_age = [age_transformer(input_image.cpu()).to('cuda')]
                input_image_age = torch.stack(input_image_age)
                result_tensor = SamNode.run_on_batch(input_image_age, net)[0]
                result_tensor = tensor2tensor(result_tensor)
                
                os.remove(f'{imgid}.jpg')
        return (result_tensor,)


if __name__ == "__main__":
    sn = SamNode()
    image_in = sys.argv[1]
    result = sn.age(image_in)
    pprint.pprint(result)
    result[0].show()
    result[0].save("result.jpg")
