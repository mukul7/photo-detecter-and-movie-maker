# A Google Photos like movie maker for your photos!

## Installation

Add Dependencies

```bash
pip3 install cv2 keras PIL tenserflow, sklearn numpy matplotlib pandas autopeg8 opencv-python
```

or

```bash
python3 -m pip install cv2, keras, PIL, tenserflow, sklearn numpy matplotlib pandas autopeg8 opencv-python
```

## To Group and generate video of random group run:

```bash
python3 main.py
```

# How it works?

Feature Extraction:
VGG16 is a convolutional neural network (CNN), that is considered avant-garde for Image recognition Tasks. This is used for feature extraction.

The Data: I am using flowers dataset, that has 210 pictures of flowers of 10 different species.

Image Processing: Pillow is used for loading images and resizing all to make a video.

For feature recognition Final layer is removed from the the VGG16 model, so that I can pass out our image in the prediction layer.

Dimensionality Reduction: PCA is used to reduce the complexity as the feature vector contains around 4000 Dimensions. You can read more about it [here](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c).

Clustering: As now I have our Image features ready. I use KMANS algorithm to cluster image according to features in K types. Here I have 10 species so K is 10.

Pickle: It is used to dump the groups data to a file so I don't have to re cluster our images.

I have got the image groups now.

Video Creation: cv2 is used to process images into video. A random group is selected from the dump I just created. As The video is 24fps. I write 48 frames to create a static video for 2 seconds. For animation I have normalised to skip frames as such all the animations takes 1 seconds.

Itertools is used to create animation.
Random is used to select random animations between frames and random group.

## Demo Video:

<figure class="video_container">
  <video controls="true" allowfullscreen="true">
    <source src="./mygeneratedvideo.mp4" type="video/mp4">
  </video>
</figure>
