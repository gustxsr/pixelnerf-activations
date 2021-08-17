import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

import util
import torch
import numpy as np
from model import make_model
from render import NeRFRenderer
import torchvision.transforms as T
import tqdm
import imageio
from PIL import Image

from sklearn.svm import SVC
from sklearn.decomposition import IncrementalPCA
from sklearn import decomposition
import pickle as pkl

def extra_args(parser):

    parser.add_argument("--size", type=int, default=128, help="Input image maxdim")
    parser.add_argument("--focal", type=float, default=131.25, help="Focal length")
    parser.add_argument(
        "--out_size",
        type=str,
        default="128",
        help="Output image size, either 1 or 2 number (w h)",
    )
    parser.add_argument("--radius", type=float, default=1.3, help="Camera distance")
    parser.add_argument("--z_near", type=float, default=0.8)
    parser.add_argument("--z_far", type=float, default=1.8)

    parser.add_argument(
        "--elevation",
        "-e",
        type=float,
        default=0.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=1,
        help="Number of video frames (rotated views)",
    )
    
    parser.add_argument("--dimension", type=int, default=0, help="Dimension of the layer number to compute activation")
    parser.add_argument("--num_comp", type=int, default=256, help="Number of PCA Components")
    parser.add_argument("--var", type=int, default=0, help="Variation to collect images from (0, 1)")
    
    return parser


class PixelNerf():
    def __init__(self):
        
        self.all_activations = []
        
        self.args, self.conf = util.args.parse_args(extra_args, default_expname="srn_car", default_data_format="srn")
        self.args.resume = True
        self.device = util.get_cuda(self.args.gpu_id[0])
        self.num_views, self.direction = 1, 0
        
        self.build_model()
        
    def build_model(self):
        
        self.net = make_model(self.conf["model"], compute_activation = True, dimension = self.args.dimension).to(device=self.device).load_weights(self.args)
        
        self.renderer = NeRFRenderer.from_conf(
            self.conf["renderer"], eval_batch_size=self.args.ray_batch_size, compute_activation = True
        ).to(device=self.device)
        
        self.render_par = self.renderer.bind_parallel(self.net, self.args.gpu_id, simple_output=True).eval()
        
    def get_components(self):
        
        z_near, z_far = self.args.z_near, self.args.z_far
        focal = torch.tensor(self.args.focal, dtype=torch.float32, device=self.device)

        in_sz = self.args.size
        sz = list(map(int, self.args.out_size.split()))
        
        if len(sz) == 1:
            H = W = sz[0]
        else:
            assert len(sz) == 2
            W, H = sz
        
        _coord_to_blender = util.coord_to_blender()
        _coord_from_blender = util.coord_from_blender()

        print("Generating rays")
        render_poses = torch.stack(
            [
                _coord_from_blender @ util.pose_spherical(angle, self.args.elevation, self.args.radius)
                #  util.pose_spherical(angle, args.elevation, args.radius)
                for angle in np.linspace(-180, 180, self.num_views + 1)[:-1]
            ],
            0,
        )  # (NV, 4, 4)

        render_rays = util.gen_rays(render_poses, W, H, focal, z_near, z_far).to(device=self.device)

        print("Render shape:", render_rays.shape)
        
        cam_pose = torch.eye(4, device=self.device)
        cam_pose[2, -1] = self.args.radius
        print("SET DUMMY CAMERA")
        
        return render_rays, in_sz, focal, cam_pose, H, W

    def compute_activation(self, layer_num, inputs, IMG_PTH):
        
        if type(IMG_PTH) == list:
            if len(IMG_PTH) != len(inputs):
                print("Check your image paths!")
                return
            
        print("Computing activations of layer", layer_num)
        
        self.all_activations = []
        render_rays, in_sz, focal, cam_pose, H, W = self.get_components()
        image_to_tensor = util.get_image_to_tensor_balanced()

        with torch.no_grad():
            for i, image_name in enumerate(inputs):
                print("IMAGE", i + 1, "of", len(inputs), "@", image_name, flush = True) 
                
                if type(IMG_PTH) == list:
                    image_path = os.path.join(IMG_PTH[i], image_name)
                else:
                    image_path = os.path.join(IMG_PTH, image_name)
                    
                image = Image.open(image_path).convert("RGB")
                image = T.Resize(in_sz)(image)
                image = image_to_tensor(image).to(device=self.device)

                self.net.encode(
                    image.unsqueeze(0), cam_pose.unsqueeze(0), focal,
                )

                for rays in tqdm.tqdm(torch.split(render_rays.view(-1, 8)[H*W*self.direction:H*W*(self.direction+1), :], 80000, dim=0)):
                    rgb, _depth = self.render_par(rays[None], layer_num = layer_num)

                if layer_num == -1:
                    layer = [activation.cpu().data.numpy() for activation in self.net.input_activation]
                    layer = np.concatenate(layer).reshape(H, W, 64, 554)

                else:
                    layer = np.concatenate(self.net.mlp_coarse.activations[layer_num]).reshape(H, W, 64, 512)

                self.net.input_activation = []
                self.net.mlp_coarse.activations = dict()

                self.all_activations.append(layer)
                
        print("Activations collected in all_activations")
