import argparse
import errno
import os
import random
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from facenet_pytorch import MTCNN, InceptionResnetV1
from models.iresnet import get_model
from omegaconf import OmegaConf
from PIL import Image
from samplers.ddim_with_grad import DDIMSamplerWithGrad
from torch.utils import data
from torchvision import transforms, utils
from util import instantiate_from_config, OptimizerDetails


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, data_aug=False, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        random.shuffle(self.paths)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((self.image_size, self.image_size), resample=PIL.Image.LANCZOS)

        return self.transform(img)


def return_cv2(img, path):
    black = [255, 255, 255]
    img = (img + 1) * 0.5
    utils.save_image(img, path, nrow=1)
    img = cv2.imread(path)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
    return img



def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


class FaceRecognition(nn.Module):
    def __init__(self, ground_truth_path, fr_crop=False, mtcnn_face=False):
        super().__init__()
        self.resnet1 = InceptionResnetV1(pretrained='vggface2').eval()
        self.resnet2 = InceptionResnetV1(pretrained='casia-webface').eval()
        self.resnet3 = get_model('r50', fp16=False)
        self.resnet3.load_state_dict(torch.load('models/weights/ms2mv3_r50.pth'))
        self.resnet3.eval()
        self.mtcnn = MTCNN(image_size=112, device='cuda')
        self.crop = fr_crop
        self.output_size = 112
        self.mtcnn_face = mtcnn_face

        # Load and preprocess ground truth image
        self.ground_truth = self.load_and_preprocess(ground_truth_path)

    def load_and_preprocess(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(image)
        image = TF.resize(image, (256, 256), interpolation=TF.InterpolationMode.BICUBIC).unsqueeze(0)
        image = self.get_faces(image, mtcnn_face=self.mtcnn_face)
        image = self.preprocess(image)
        # Ensure the output is a 4D tensor (batch, channels, height, width)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image

    def preprocess(self, image):
        if isinstance(image, torch.Tensor):
            # If input is already a tensor, assume it's in the range [-1, 1]
            image = (image + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
            image = image.clamp(0, 1)

            # Ensure the tensor is 4D (batch, channels, height, width)
            if image.dim() == 5:
                image = image.squeeze(0)  # Remove the extra dimension
            elif image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dimension
        else:
            # If input is a PIL Image
            transform = transforms.Compose([
                transforms.Resize((self.output_size, self.output_size)),
                transforms.ToTensor(),
            ])
            image = transform(image)

        # Normalize
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return normalize(image)

    def extract_face(self, imgs, batch_boxes, mtcnn_face=False):
        image_size = imgs.shape[-1]
        faces = []
        for i in range(imgs.shape[0]):
            img = imgs[i]
            if not mtcnn_face:
                box = [48, 48, 208, 208]
                crop_face = img[None, :, box[1]:box[3], box[0]:box[2]]
            elif batch_boxes[i] is not None:
                box = batch_boxes[i][0]
                margin = [
                    self.mtcnn.margin * (box[2] - box[0]) / (self.output_size - self.mtcnn.margin),
                    self.mtcnn.margin * (box[3] - box[1]) / (self.output_size - self.mtcnn.margin),
                ]

                box = [
                    int(max(box[0] - margin[0] / 2, 0)),
                    int(max(box[1] - margin[1] / 2, 0)),
                    int(min(box[2] + margin[0] / 2, image_size)),
                    int(min(box[3] + margin[1] / 2, image_size)),
                ]
                crop_face = img[None, :, box[1]:box[3], box[0]:box[2]]
            else:
                return None

            faces.append(F.interpolate(crop_face, size=self.output_size, mode='bicubic'))
        new_faces = torch.cat(faces)

        return (new_faces - 127.5) / 128.0

    def get_faces(self, x, mtcnn_face=False):
        img = (x + 1.0) * 0.5 * 255.0
        img = img.permute(0, 2, 3, 1)
        with torch.no_grad():
            batch_boxes, batch_probs, batch_points = self.mtcnn.detect(img, landmarks=True)
            # Select faces
            batch_boxes, batch_probs, batch_points = self.mtcnn.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.mtcnn.selection_method
            )

        img = img.permute(0, 3, 1, 2)
        faces = self.extract_face(img, batch_boxes, mtcnn_face)
        return faces

    def forward(self, x, return_faces=False, mtcnn_face=None):
        x = TF.resize(x, (256, 256), interpolation=TF.InterpolationMode.BICUBIC)

        if mtcnn_face is None:
            mtcnn_face = self.mtcnn_face

        faces = self.get_faces(x, mtcnn_face=mtcnn_face)
        if faces is None:
            print("No faces detected")
            return faces

        if not self.crop:
            out = self.resnet1(x)
        else:
            out = self.resnet1(faces)

        if return_faces:
            return out, faces
        else:
            return out

    def cal_loss(self, image, timestep, image_number, base_folder):
        x = TF.resize(image, (256, 256), interpolation=TF.InterpolationMode.BICUBIC)

        processed_image = self.get_faces(x, mtcnn_face=self.mtcnn_face)
        
        intermediate_folder = os.path.join(base_folder, 'intermediates', f'image{image_number}')
        os.makedirs(intermediate_folder, exist_ok=True)
        
        utils.save_image(processed_image, f'{intermediate_folder}/img_at_step_{500 - timestep}.png')

        if self.ground_truth.dim() == 3:
            self.ground_truth = self.ground_truth.unsqueeze(0)
        if processed_image.dim() == 3:
            processed_image = processed_image.unsqueeze(0)

        processed_image = processed_image.requires_grad_(True)

        with torch.no_grad():
            gt_emb1 = self.resnet1(self.ground_truth)
            gt_emb2 = self.resnet2(self.ground_truth)
            gt_emb3 = self.resnet3(self.ground_truth)

        img_emb1 = self.resnet1(processed_image)
        img_emb2 = self.resnet2(processed_image)
        img_emb3 = self.resnet3(processed_image)

        # Calculate losses
        dist1 = F.cosine_similarity(gt_emb1, img_emb1, dim=1)
        dist2 = F.cosine_similarity(gt_emb2, img_emb2, dim=1)
        dist3 = F.cosine_similarity(gt_emb3, img_emb3, dim=1)

        print(f"Timestep {500 - timestep}: Loss for first model: {dist1.item()}")
        print(f"Timestep {500 - timestep}: Loss for second model: {dist2.item()}")
        print(f"Timestep {500 - timestep}: Loss for third model: {dist3.item()}")

        if hasattr(self, 'fr_model'):
            if self.fr_model == 1:
                loss = (10 * (1 - dist1)) ** 2 + .25 * (10 * dist2) ** 2 + .25 * (10 * dist3) ** 2
            elif self.fr_model == 2:
                loss = (10 * (1 - dist2)) ** 2 + .25 * (10 * dist1) ** 2 + .25 * (10 * dist3) ** 2
            elif self.fr_model == 3:
                loss = (10 * (1 - dist3)) ** 2 + .25 * (10 * dist1) ** 2 + .25 * (10 * dist2) ** 2
        else:
            loss = (10 * (1 - dist2)) ** 2 + .25 * (10 * dist1) ** 2 + .25 * (10 * dist3) ** 2
            
        return loss

    def cuda(self):
        self.resnet1 = self.resnet1.cuda()
        self.resnet2 = self.resnet2.cuda()
        self.resnet3 = self.resnet3.cuda()
        self.mtcnn = self.mtcnn.cuda()
        self.ground_truth = self.ground_truth.cuda()
        return self




def get_optimation_details(args):
    mtcnn_face = not args.center_face
    print('mtcnn_face')
    print(mtcnn_face)

    guidance_func = FaceRecognition(args.input_image, fr_crop=args.fr_crop, mtcnn_face=mtcnn_face).cuda()
    guidance_func.fr_model = args.fr_model  # Add the fr_model parameter
    operation = OptimizerDetails()

    operation.num_steps = args.optim_num_steps
    operation.operation_func = guidance_func

    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff

    operation.guidance_3 = args.optim_forward_guidance
    operation.guidance_2 = args.optim_backward_guidance

    operation.optim_guidance_3_wt = args.optim_forward_guidance_wt
    operation.original_guidance = args.optim_original_conditioning

    operation.warm_start = args.optim_warm_start
    operation.print = args.optim_print
    operation.print_every = 5
    operation.folder = args.optim_folder

    return operation


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sd_1-4.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument("--optim_lr", default=1e-2, type=float)
    parser.add_argument('--optim_max_iters', type=int, default=1)
    parser.add_argument('--optim_mask_type', type=int, default=1)
    parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
    parser.add_argument('--optim_forward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_backward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_original_conditioning', action='store_true', default=False)
    parser.add_argument("--optim_forward_guidance_wt", default=5.0, type=float)
    parser.add_argument("--optim_tv_loss", default=None, type=float)
    parser.add_argument('--optim_warm_start', action='store_true', default=False)
    parser.add_argument('--optim_print', action='store_true', default=False)
    parser.add_argument('--optim_folder', default='./temp/')
    parser.add_argument("--optim_num_steps", nargs="+", default=[1], type=int)
    parser.add_argument('--text_type', type=int, default=1)
    parser.add_argument("--text", required=True)
    parser.add_argument("--negative_prompt", required=False)
    parser.add_argument('--input_image', type=str, required=True, help="Path to the input image")
    parser.add_argument('--fr_crop', action='store_true')
    parser.add_argument('--center_face', action='store_true')
    parser.add_argument("--trials", default=20, type=int)
    parser.add_argument(
        "--fr_model",
        type=int,
        default=2,
        help="Face recognition model to optimize (1, 2, or 3)",
    )

    opt = parser.parse_args()
    results_folder = opt.optim_folder
    create_folder(results_folder)

    set_seed(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[device])
    model.eval()

    sampler = DDIMSamplerWithGrad(model)
    operation = get_optimation_details(opt)

    image_size = 256
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    # Load and process the single input image
    input_image = Image.open(opt.input_image).convert('RGB')
    og_img = transform(input_image).unsqueeze(0).to(device)
    og_img = og_img.cuda()

    torch.set_grad_enabled(False)

    prompt = opt.text
    print(prompt)

    # Save the original image
    temp = (og_img + 1) * 0.5
    utils.save_image(temp, f'{results_folder}/og_img.png')

    with torch.no_grad():
        og_img_guide, og_img_mask = operation.operation_func(og_img, return_faces=True, mtcnn_face=True)
        utils.save_image((og_img_mask + 1) * 0.5, f'{results_folder}/og_img_cut.png')
        print(f"Image saved at {results_folder}/og_img_cut.png")

    c = model.module.get_learned_conditioning([prompt])
    uc = model.module.get_learned_conditioning([""])  # unconditional (empty) prompt

    if opt.negative_prompt:
        n = model.module.get_learned_conditioning([opt.negative_prompt])  # negative prompt
        # Concatenate all three conditions
        c_in = torch.cat([uc, n, c])
    else:
        # Original behavior with just unconditional and conditional
        c_in = torch.cat([uc, c])

    for multiple_tries in range(opt.trials):
        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
        samples_ddim, start_zt = sampler.sample(
            S=opt.ddim_steps,
            conditioning=c_in,
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=opt.scale,
            unconditional_conditioning=uc,
            eta=opt.ddim_eta,
            operated_image=og_img_guide,
            operation=operation,
            identifier=multiple_tries
        )

        x_samples_ddim = model.module.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        utils.save_image(x_samples_ddim, f'{results_folder}/new_img_{multiple_tries}.png')

if __name__ == "__main__":
    main()




