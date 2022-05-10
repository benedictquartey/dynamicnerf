import json
import argparse
import pathlib

parser = argparse.ArgumentParser(prog="pipeline", description='...')
parser.add_argument('--expname', default='', type=str)
args = parser.parse_args()

json_file = pathlib.Path(f"data/nerf_llff_data/{args.expname}/transforms.json")
transforms_data = json.load(open(json_file, "r"))
new_transforms = dict()

for k in transforms_data.keys():
    if k in ['frames', "camera_angle_x"]:
        new_transforms[k] = transforms_data[k]

new_transforms['frames'].sort(key=lambda x: int(pathlib.Path(x["file_path"]).stem))

for t, frame in enumerate(new_transforms['frames']):
    del frame['sharpness']
    frame['rotation'] = 0.06283185307179587
    frame['time'] = float(t)/(len(new_transforms['frames'])-1)

json.dump(new_transforms, open(f"data/nerf_llff_data/{args.expname}/transforms_train.json", "w+"))
json.dump(new_transforms, open(f"data/nerf_llff_data/{args.expname}/transforms_val.json", "w+"))
json.dump(new_transforms, open(f"data/nerf_llff_data/{args.expname}/transforms_test.json", "w+"))