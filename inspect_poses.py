import os
import imageio
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange


from run_dnerf_helpers import *

from load_llff import load_llff_data
from load_blender import load_blender_data

try:
    from apex import amp
except ImportError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False



# print("loading llff")
# llff_images, llff_poses, llff_bds, llff_render_poses, llff_i_test = load_llff_data("data/nerf_llff_data/llff_yong_kati",
#                                                          factor=8,recenter=True, bd_factor=.75, spherify=True)

# print("loading blender data ...\n")
# blndr_images, blndr_poses, blndr_times, blndr_render_poses, blndr_render_times, blndr_hwf, blndr_i_split = load_blender_data("data/nerf_llff_data/yong_kati_2", 
#                                                                                                                                 half_res=False, testskip=1)

# print(".......LLFF Data.......")
# print(f"images: {llff_images.shape}")
# print(f"poses: {llff_poses.shape}")
# print(f"bds: {llff_bds.shape}")
# print(f"render_poses: {llff_render_poses.shape}")
# print(f"i_test: {np.asarray(llff_i_test).shape}")

# print("\n.......BLNDR Data.......")
# print(f"images: {blndr_images.shape}")
# print(f"poses: {blndr_poses.shape}")
# print(f"blndr_times: {blndr_times.shape}")
# print(f"blndr_render_times: {blndr_render_times.shape}")
# print(f"blndr_hwf: {np.asarray(blndr_hwf).shape}")
# print(f"render_poses: {blndr_render_poses.shape}")
# print(f"i_test: {np.asarray(blndr_i_split).shape}")



def check_llff():
    #BEN_MOD_COMMENTS || what is missing is times data and render_times data, add those for llff and run code
    llffhold = 8
    datadir = "data/nerf_llff_data/llff_yong_kati"
    no_ndc = True
    images, poses, bds, render_poses, i_test = load_llff_data(datadir,
                                                            factor=8,recenter=True, bd_factor=.75, spherify=True)
    ##adding times
    times = np.zeros((images.shape[0]))
    for t in range(len(images)):
        times[t] = float(t)/(len(images)-1)

    render_times = torch.linspace(0., 1., render_poses.shape[0])
    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]
    # print('Loaded llff', images.shape, render_poses.shape, hwf, datadir)
    # print("times:", times, times.shape)
    # print("render_times:", render_times, render_times.shape)
    if not isinstance(i_test, list):
        i_test = [i_test]

    if llffhold > 0:
        print('Auto LLFF holdout,', llffhold)
        i_test = np.arange(images.shape[0])[::llffhold]
        # print("itest",i_test)

    i_val = i_test
    # i_train = np.array([i for i in np.arange(int(images.shape[0])) if
    #                 (i not in i_test and i not in i_val)])

    i_train = np.array([i for i in np.arange(int(images.shape[0]))])

    print('DEFINING BOUNDS')
    if no_ndc:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
        
    else:
        near = 0.
        far = 1.

    print('NEAR FAR', near, far)

    print(".......LLFF Data.......")
    print(f"images: {images.shape}")
    print(f"poses: {poses.shape}")
    print(f"bds: {bds.shape}")
    print(f"render_poses: {render_poses.shape}")
    print(f"i_test: {np.asarray(i_test).shape}")


def check_blender():
    datadir = "data/nerf_llff_data/yong_kati_2"
    half_res = False
    testskip = 1
    white_bkgd = False

    images, poses, times, render_poses, render_times, hwf, i_split = load_blender_data(datadir, half_res, testskip)
    print('Loaded blender', images.shape, render_poses.shape, hwf, datadir)
    # print('time: ')
    # print(*times, sep=", ")
    # print('render_time: ', render_times)
    i_train, i_val, i_test = i_split

    near = 2.
    far = 6.

    if white_bkgd:
        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    else:
        images = images[...,:3]

    # images = [rgb2hsv(img) for img in images]

    print("\n.......BLNDR Data.......")
    print(f"images: {images.shape}")
    print(f"poses: {poses.shape}")
    print(f"blndr_times: {times.shape}")
    print(f"blndr_render_times: {render_times.shape}")
    print(f"blndr_hwf: {np.asarray(hwf).shape}")
    print(f"render_poses: {render_poses.shape}")
    print(f"i_test: {np.asarray(i_split).shape}")


# print("i_train:", i_train)
# min_time, max_time = times[i_train[0]], times[i_train[-1]]
# print(f"min_time: {min_time} || max_time: {max_time}")
# assert min_time == 0., "time must start at 0"
# assert max_time == 1., "max time must be 1"


# # Cast intrinsics to right types
# H, W, focal = hwf
# H, W = int(H), int(W)
# hwf = [H, W, focal]

# #BEN_MOD_COMMENTS necessary???
# if K is None:
#     K = np.array([
#         [focal, 0, 0.5*W],
#         [0, focal, 0.5*H],
#         [0, 0, 1]
#     ])



check_llff()
check_blender()