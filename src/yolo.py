import os
import torch

import yolov5


def yolo8_model(img):
    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

    # Inference
    results = model(img)

    return results


def yolo_model(img):
    # load pretrained model
    model = yolov5.load('yolov5s.pt')

    # # set model parameters
    # model.conf = 0.25  # NMS confidence threshold
    # model.iou = 0.45  # NMS IoU threshold
    # model.agnostic = False  # NMS class-agnostic
    # model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1  # maximum number of detections per image

    # perform inference
    results = model(img)

    return results



if __name__ == "__main__":
    img = os.path.join(os.path.expanduser('~'), "ltl_transfer/fetch/multiobj/juice/images/hand_color_image_0100.jpg")  # or file, Path, PIL, OpenCV, numpy, list
    results = yolo_model(img)

    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    results.show()

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    breakpoint()


    # img_dpath = os.path.join(os.path.expanduser('~'), "ltl_transfer/fetch/multiobj/book/images")
    # detect_dpath = os.path.join(os.path.expanduser('~'), "ltl_transfer/fetch/multiobj/book/output_yolo")
    # # os.makedirs(detect_dpath, exist_ok=True)

    # img_fnames = os.listdir(img_dpath)
    # for img_fname in img_fnames:
    #     results = yolo_model(os.path.join(img_dpath, img_fname))

    #     # save results into "results/" folder
    #     results.save(save_dir=f"{detect_dpath}")
    #     # mkdir fetch/multiobj/juice/yolo_output/
    #     # mv fetch/multiobj/book/output_yolo*/*.jpg fetch/multiobj/book/yolo_output/
    #     # rm -r fetch/multiobj/book/output_yolo*
