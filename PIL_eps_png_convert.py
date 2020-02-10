from PIL import Image
import argparse
import glob
from PIL import Image
import os

parser = argparse.ArgumentParser(description='Convert a folder of EPS files to .PNG files')
parser.add_argument('--folder', type=str, default='images')
parser.add_argument('--output_folder', type=str, default='images_converted')
args = parser.parse_args()

if not os.path.isdir(args.output_folder):
    os.mkdir(args.output_folder)

for file in glob.glob(args.folder+"\\*.eps"):
    file_name = file[len(args.folder)+1:-len(".eps")]
    im = Image.open(file)
    im.save(args.output_folder + "\\" + file_name+".png")
