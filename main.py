# coding: utf-8
import os
import sys
import random

import cv2
import PIL
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import tqdm





from sklearn.cluster import KMeans


def createVideoSummary(videoFile,
              n_keypoints=10,
              n_samples=100,
              range_frames=60,
              outfile='test.avi',
              fps=30,
              codecs='H264',
              use_cuda=None,
              ):

    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    n_half_range = int(range_frames/2)


    # feature extractor
    resnet = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(resnet.children())[:-1])
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    if use_cuda:
        model = model.cuda()

    # Open the video file
    video_caps = cv2.VideoCapture(videoFile)
    height = video_caps.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = video_caps.get(cv2.CAP_PROP_FRAME_WIDTH )
    print height, width


    n_all_frames = int(video_caps.get(cv2.CAP_PROP_FRAME_COUNT))
    set_sample_points = set([random.randint(0, n_all_frames) for i in range(n_samples)])
    d_map_keypoint_range = {}

    for p in set_sample_points:
        durations = list(range(max(0, p-n_half_range), min(p+n_half_range, n_all_frames)))
        d_map_keypoint_range[p] = durations

    lst_features = []
    lst_frame_nums = []

    # d_keeped_frames = {}

    print('processing frames')
    for f in tqdm.trange(n_all_frames):

        ret, _frame = video_caps.read()

        if f not in set_sample_points:
            continue

        if ret:

            frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
            frame = PIL.Image.fromarray(frame)
            frame = transformer(frame)
            frame = frame.unsqueeze(0).float()
            with torch.no_grad():
                if use_cuda:
                    frame = frame.cuda()
                feature = model(frame)
                if use_cuda:
                    feature = feature.cpu()

            lst_features += [feature.reshape(1, -1)]
            lst_frame_nums += [f]
    video_caps.release()


    features = torch.cat(lst_features)

    # clustering

    X = features.numpy()
    kmeans = KMeans(n_clusters=n_keypoints, random_state=0).fit(X)
    distances = kmeans.transform(X)
    indices = np.argmin(distances, axis=0).tolist()
    indices.sort()

    keypoints = [lst_frame_nums[f] for f in indices]

    set_selected_frames = set([ f for p in keypoints for f in d_map_keypoint_range[p]])


    # Open the video file
    video_caps = cv2.VideoCapture(videoFile)
    writer = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))
    print('Writing a file')

    for f in tqdm.trange(n_all_frames):

        ret, frame = video_caps.read()

        if f not in set_selected_frames:
            continue

        if ret:
            writer.write(frame)

    writer.release()
    video_caps.release()

if __name__ == '__main__':

    path_file = './Mandelbrot Zoom 10^227 [1080x1920]-PD2XgQOyCCk.mp4'
    createVideoSummary(path_file)


