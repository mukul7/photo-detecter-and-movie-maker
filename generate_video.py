import pickle
import os
import random
import cv2
from PIL import Image
from cv2 import VideoWriter_fourcc
import itertools


ANIMATION_TIME = 1
FRAMES_PER_SEC = 24
IMAGE_STILL_TIME = 2  # Image will show for 2 seconds after animation
STILL_FRAME_COUNT = IMAGE_STILL_TIME * FRAMES_PER_SEC

CURRENT_DIR = os.getcwd()
IMAGE_FOLDER = "archive/flower_images/flower_images"
RESIZED_IMAGES = "archive/resized_images"

vidCodec = VideoWriter_fourcc(*"mp4v")


def stillFrame(video, frame, n_frame):
    for i in range(n_frame):
        video.write(frame)


def findModular(frames):
    return int((frames/FRAMES_PER_SEC)/ANIMATION_TIME)


def getRandomMusic():
    return random.choice((os.listdir(os.path.join(CURRENT_DIR, 'archive/music'))))


def getFrameWithAnimation(video, initial_frame, final_frame):
    animation = random.randrange(0, 4)
    frame_count = 0
    # Animation Start revealing from top
    if animation == 0:
        modular = findModular(len(final_frame))
        for y in range(len(final_frame)):
            for x in range(len(final_frame[y])):
                initial_frame[y][x] = final_frame[y][x]
            if(frame_count % modular == 0):
                video.write(initial_frame)
            frame_count += 1
    # Drop from top
    elif animation == 1:
        modular = findModular(len(final_frame))
        for y in range(len(final_frame)):
            cc = 0
            if(y > 0):
                for row in final_frame[-y:]:
                    initial_frame[cc] = row
                    cc += 1
            if(frame_count % modular == 0):
                video.write(initial_frame)
            frame_count += 1
    # Fade In
    elif animation == 2:
        frame_count = 0
        comblist = list(itertools.product(
            *[range(len(final_frame)), range(len(final_frame[0]))]))
        random.shuffle(comblist)
        modular = findModular(len(comblist))

        for y, x in comblist:
            initial_frame[y][x] = final_frame[y][x]
            if(frame_count % modular == 0):
                video.write(initial_frame)
            frame_count += 1
    # Horizonatal Blinds
    elif animation == 3:
        ylist = list(range(len(final_frame)))
        random.shuffle(ylist)

        modular = findModular(len(final_frame))

        for y in ylist:
            initial_frame[y] = final_frame[y]
            if(frame_count % modular == 0):
                video.write(initial_frame)
            frame_count += 1


def generate_video(imagesList, height, width):
    resized_image_folder = os.path.join(CURRENT_DIR, RESIZED_IMAGES)
    video_name = 'mygeneratedvideo.mp4'

    images = imagesList
    random.shuffle(images)

    # Initialize 1st frame with 1st Image
    frame = cv2.imread(os.path.join(resized_image_folder, images[0]))
    video = cv2.VideoWriter(video_name, vidCodec,
                            float(FRAMES_PER_SEC), (width, height))  # 24FPS

    stillFrame(video, frame, STILL_FRAME_COUNT)
    for image in images[1:]:
        getFrameWithAnimation(video, frame, cv2.imread(
            os.path.join(resized_image_folder, image)))
        stillFrame(video, frame, STILL_FRAME_COUNT)

    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated

    bg_music = os.path.join(CURRENT_DIR, 'archive/music', getRandomMusic())
    os.system(
        "ffmpeg -i "+video_name+" -i " + bg_music + " -map 0 -map 1 -codec copy -shortest " + video_name)


with open(os.path.join(CURRENT_DIR, 'dump', 'config.groups'), 'rb') as config_dictionary_file:

    config_dictionary = pickle.load(config_dictionary_file)
    # Select Images of a random group
    images = config_dictionary[random.randrange(0, len(config_dictionary))]

    mean_height = 0
    mean_width = 0

    num_of_images = len(images)

    for file in images:
        im = Image.open(os.path.join(CURRENT_DIR, IMAGE_FOLDER, file))
        width, height = im.size
        mean_width += width
        mean_height += height

    # Finding the mean height and width of all images.
    # Images of only this size can be added to video
    mean_width = int(mean_width / num_of_images)
    mean_height = int(mean_height / num_of_images)

    # Resizing images to mean height and width
    for file in images:
        if file.endswith("png"):
            im = Image.open(os.path.join(CURRENT_DIR, IMAGE_FOLDER, file))
            # resizing
            imResize = im.resize((mean_width, mean_height), Image.LANCZOS)
            imResize.save(os.path.join(CURRENT_DIR, RESIZED_IMAGES,
                          file), 'png', quality=100)  # setting quality

    generate_video(images, height, width)
