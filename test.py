import argparse
import glob
import os
import PIL.Image as pil
from PIL import Image
import cv2
import numpy as np
import torch

from monolayout import model
from torchvision import transforms
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(
        description="Testing arguments for MonoLayout")
    parser.add_argument("--video_path", type=str, 
      help="path to folder of video", required=True)
    parser.add_argument("--model_path", type=str, 
      help="path to MonoLayout model", required=True)
    parser.add_argument("--out_dir", type=str, 
      help="output directory to save topviews")
    parser.add_argument("--type", type=str, 
      default="static/dynamic/both")
    return parser.parse_args()


def save_topview(tv, original_width, original_height):
    tv_np = tv.squeeze().cpu().numpy()
    true_top_view = np.zeros((tv_np.shape[1], tv_np.shape[2]), dtype=np.uint8)
    true_top_view[tv_np[1] > tv_np[0]] = 255
    true_top_view = cv2.resize(true_top_view, 
      (original_width, original_height),  
        interpolation = cv2.INTER_NEAREST) 
    return true_top_view


def test(args):
    models = {}
    device = torch.device("cuda")
    encoder_path = os.path.join(args.model_path, "encoder.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    feed_height = encoder_dict["height"]
    feed_width = encoder_dict["width"]
    models["encoder"] = model.Encoder(18, feed_width, feed_height, False)
    filtered_dict_enc = {
      k: v for k, 
      v in encoder_dict.items() if k in models["encoder"].state_dict()}
    models["encoder"].load_state_dict(filtered_dict_enc)

    cam = cv2.VideoCapture(args.video_path)
    _, img = cam.read()
    
    if img is None:
        print("Can not find args.video_path: {}".format(args.video_path))
  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(img)
    original_width, original_height = input_image.size 
    total_frames = 1
    
    while True:
        _, img = cam.read()
        if img is None:
            break
        total_frames += 1

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.type == "both":
        static_decoder_path = os.path.join(
          args.model_path, "static_decoder.pth")
        dynamic_decoder_path = os.path.join(
          args.model_path, "dynamic_decoder.pth")
        models["static_decoder"] = model.Decoder(
          models["encoder"].resnet_encoder.num_ch_enc)
        models["static_decoder"].load_state_dict(
          torch.load(static_decoder_path, map_location=device))
        models["dynamic_decoder"] = model.Decoder(
          models["encoder"].resnet_encoder.num_ch_enc)
        models["dynamic_decoder"].load_state_dict(
          torch.load(dynamic_decoder_path, map_location=device))
        static_out = cv2.VideoWriter(
          os.path.join(args.out_dir, "static.avi"),
            fourcc, 30, (2 * original_width, original_height))
        dynamic_out = cv2.VideoWriter(
          os.path.join(args.out_dir, "dynamic.avi"),
            fourcc, 22, (2 * original_width, original_height))
    else:
        decoder_path = os.path.join(args.model_path, "decoder.pth")
        models["decoder"] = model.Decoder(
          models["encoder"].resnet_encoder.num_ch_enc)
        models["decoder"].load_state_dict(
          torch.load(decoder_path, map_location=device))
        normal_out = cv2.VideoWriter(
          os.path.join(args.out_dir, "normal.avi"),
            fourcc, 22, (2 * original_width, original_height))

    for key in models.keys():
        models[key].to(device)
        models[key].eval()
    
    cam = cv2.VideoCapture(args.video_path)
    pbar = tqdm(total=total_frames)
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        while True:
            _, img = cam.read()
            if img is None:
                break

            # Load image and preprocess
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(img)
            original_width, original_height = input_image.size
            input_image = input_image.resize(
              (feed_width, feed_height), pil.LANCZOS)
            image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            image = image.to(device)
            features = models["encoder"](image)
            
            if args.type == "both":
                static_tv = models["static_decoder"](
                  features, is_training=False)
                dynamic_tv = models["dynamic_decoder"](
                  features, is_training=False)
                true_static_top_view = save_topview(static_tv, 
                  original_width, original_height)
                true_dynamic_top_view = save_topview(dynamic_tv, 
                  original_width, original_height)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                rows_rgb, cols_rgb, channels = img.shape
                rows_gray, cols_gray = true_static_top_view.shape
                rows_comb = max(rows_rgb, rows_gray)
                cols_comb = cols_rgb + cols_gray
                comb = np.zeros(
                  shape=(rows_comb, cols_comb, channels), dtype=np.uint8)
                comb[:rows_rgb, :cols_rgb] = img
                comb[:rows_gray, cols_rgb:] = true_static_top_view[:, :, None]
                static_out.write(comb)

                rows_gray, cols_gray = true_static_top_view.shape
                rows_comb = max(rows_rgb, rows_gray)
                cols_comb = cols_rgb + cols_gray
                comb = np.zeros(
                  shape=(rows_comb, cols_comb, channels), dtype=np.uint8)
                comb[:rows_rgb, :cols_rgb] = img
                comb[:rows_gray, cols_rgb:] = true_static_top_view[:, :, None]
                dynamic_out.write(comb)

            else:
                tv = models["decoder"](features, is_training=False)
                true_normal_top_view = save_topview(tv,
                  original_width, original_height)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                rows_rgb, cols_rgb, channels = img.shape
                rows_gray, cols_gray = true_normal_top_view.shape
                rows_comb = max(rows_rgb, rows_gray)
                cols_comb = cols_rgb + cols_gray
                comb = np.zeros(
                  shape=(rows_comb, cols_comb, channels), dtype=np.uint8)
                comb[:rows_rgb, :cols_rgb] = img
                comb[:rows_gray, cols_rgb:] = true_normal_top_view[:, :, None]

                normal_out.write(comb)
                pbar.update(1)
    
    print('-> Done!')


if __name__ == "__main__":
    args = get_args()
    test(args)
