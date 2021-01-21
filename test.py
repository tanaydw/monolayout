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


def get_args():
    parser = argparse.ArgumentParser(
        description="Testing arguments for MonoLayout")
    parser.add_argument("--video_path", type=str, help="path to folder of video", required=True)
    parser.add_argument("--model_path", type=str, help="path to MonoLayout model", required=True)
    parser.add_argument("--out_dir", type=str, help="output directory to save topviews")
    parser.add_argument("--type", type=str, default="static/dynamic/both")
    return parser.parse_args()


def save_topview(tv):
    tv_np = tv.squeeze().cpu().numpy()
    true_top_view = np.zeros((tv_np.shape[1], tv_np.shape[2]))
    true_top_view[tv_np[1] > tv_np[0]] = 255
    return true_top_view


def test(args):
    models = {}
    device = torch.device("cuda")
    encoder_path = os.path.join(args.model_path, "encoder.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    feed_height = encoder_dict["height"]
    feed_width = encoder_dict["width"]
    models["encoder"] = model.Encoder(18, feed_width, feed_height, False)
    filtered_dict_enc = {k: v for k, v in encoder_dict.items() if k in models["encoder"].state_dict()}
    models["encoder"].load_state_dict(filtered_dict_enc)

    cam = cv2.VideoCapture(args.video_path)
    _, img = cam.read()
    if img == None:
        print("Can not find args.video_path: {}".format(args.video_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(img)
    original_width, original_height = input_image.size
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.type == "both":
        static_decoder_path = os.path.join(args.model_path, "static_decoder.pth")
        dynamic_decoder_path = os.path.join(args.model_path, "dynamic_decoder.pth")
        models["static_decoder"] = model.Decoder(models["encoder"].resnet_encoder.num_ch_enc)
        models["static_decoder"].load_state_dict(torch.load(static_decoder_path, map_location=device))
        models["dynamic_decoder"] = model.Decoder(models["encoder"].resnet_encoder.num_ch_enc)
        models["dynamic_decoder"].load_state_dict(torch.load(dynamic_decoder_path, map_location=device))
        static_out = cv2.VideoWriter(os.path.join(args.out_dir, "static.mp4"),
            fourcc, 30, (original_width, original_height))
        dynamic_out = cv2.VideoWriter(os.path.join(args.out_dir, "dynamic.mp4"),
            fourcc, 30, (original_width, original_height))
    else:
        decoder_path = os.path.join(args.model_path, "decoder.pth")
        models["decoder"] = model.Decoder(models["encoder"].resnet_encoder.num_ch_enc)
        models["decoder"].load_state_dict(torch.load(decoder_path, map_location=device))
        normal_out = cv2.VideoWriter(os.path.join(args.out_dir, "normal.mp4"),
            fourcc, 30, (original_width, original_height))

    for key in models.keys():
        models[key].to(device)
        models[key].eval()
    
    i = 0
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        while True:
            if i != 0:
                _, img = cam.read()
                if img is None:
                    break
            
            # Load image and preprocess
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(img)
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = models["encoder"](input_image)
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            
            if args.type == "both":
                static_tv = models["static_decoder"](features, is_training=False)
                dynamic_tv = models["dynamic_decoder"](features, is_training=False)
                true_static_top_view = save_topview(static_tv)
                true_dynamic_top_view = save_topview(dynamic_tv)
                static_out.write(true_static_top_view)
                dynamic_out.write(ture_dynamic_top_view)
            else:
                tv = models["decoder"](features, is_training=False)
                true_normal_top_view = save_topview(tv)
                normal_out.write(true_normal_top_view)
            i += 1
            print(str(i) + "th frames processed.")

    print('-> Done!')


if __name__ == "__main__":
    args = get_args()
    test(args)
