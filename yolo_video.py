import os
import sys
import argparse
from testYolo import YOLO, detect_video
from PIL import Image
from glob import glob




def detect_img(yolo):
    images_dir = '..\\coco\\val2014\\images'
    image_save_dir = '..\\coco\\val2014\\predictionsV2'
    image_size = 128

    os.makedirs(image_save_dir, exist_ok=True)
    img_paths = glob(os.path.join(images_dir, '*'))

    for img_path in img_paths:
        # resize
        image = Image.open(img_path)
        rgb_im = image.convert('RGB')
        rgb_im.thumbnail([image_size,image_size])
        r_image = yolo.detect_image(image)

#        r_image.show()


        # make background
#        back_ground = Image.new("RGB", (image_size,image_size), color=(255,255,255))
#        back_ground.paste(r_image)
        # make path
        save_path = os.path.join(image_save_dir, os.path.basename(img_path))
        end_index = save_path.rfind('.')
        save_path = save_path[0:end_index]+'.txt'
        print('save',save_path)
        with open(save_path, "w") as f:
            f.write(r_image)

#        back_ground.save(save_path,quality=95,format='JPEG')
        #r_image.save(save_path, quality=95, format='JPEG')
#
#    while True:
#        img = input('Input image filename:')
#        try:
#            image = Image.open(img)
#        except:
#            print('Open Error! Try again!')
#            continue
#        else:

    yolo.close_session()


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
