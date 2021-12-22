try:
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
except:
    pass
import cv2,os
import numpy as np
import random
import torch
import PIL.Image
from PIL import Image
from torchvision.transforms import functional as F
from torch.nn import functional as F2
import math

def run_seq():
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    # if not isinstance(images,list):
    #     images=[images]

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image.
    seq = iaa.Sequential(
        [
            #
            # Apply the following augmenters to most images.
            #
            # iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            # iaa.Flipud(0.2),  # vertically flip 20% of all images

            # crop some of the images by 0-10% of their height/width
            # sometimes(iaa.Crop(percent=(0, 0.1))),
            # sometimes(iaa.Crop(percent=(0, 0.05))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                # rotate=(-45, 45),
                rotate=(-5, 5),
                # shear=(-16, 16),
                shear=(-5, 5),
                order=[0, 1],
                # cval=(0, 255),
                cval= 144, # 填充像素值
                # mode=ia.ALL # 默认常数值填充边界
            )),

            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 5),
                       [
                           # Convert some images into their superpixel representation,
                           # sample between 20 and 200 superpixels per image, but do
                           # not replace all superpixels with their average, only
                           # some of them (p_replace).
                           sometimes(
                               iaa.Superpixels(
                                   p_replace=(0, 1.0),
                                   n_segments=(20, 200)
                               )
                           ),

                           # Blur each image with varying strength using
                           # gaussian blur (sigma between 0 and 3.0),
                           # average/uniform blur (kernel size between 2x2 and 7x7)
                           # median blur (kernel size between 3x3 and 11x11).
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 11)),
                           ]),

                           # Sharpen each image, overlay the result with the original
                           # image using an alpha between 0 (no sharpening) and 1
                           # (full sharpening effect).
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                           # Same as sharpen, but for an embossing effect.
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                           # Search in some images either for all edges or for
                           # directed edges. These edges are then marked in a black
                           # and white image and overlayed with the original image
                           # using an alpha of 0 to 0.7.
                           sometimes(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0, 0.7)),
                               iaa.DirectedEdgeDetect(
                                   alpha=(0, 0.7), direction=(0.0, 1.0)
                               ),
                           ])),

                           # Add gaussian noise to some images.
                           # In 50% of these cases, the noise is randomly sampled per
                           # channel and pixel.
                           # In the other 50% of all cases it is sampled once per
                           # pixel (i.e. brightness change).
                           iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                           ),

                           # Either drop randomly 1 to 10% of all pixels (i.e. set
                           # them to black) or drop them on an image with 2-5% percent
                           # of the original size, leading to large dropped
                           # rectangles.
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout(
                                   (0.03, 0.15), size_percent=(0.02, 0.05),
                                   per_channel=0.2
                               ),
                           ]),

                           # Invert each image's channel with 5% probability.
                           # This sets each pixel value v to 255-v.
                           iaa.Invert(0.05, per_channel=True),  # invert color channels

                           # Add a value of -10 to 10 to each pixel.
                           iaa.Add((-10, 10), per_channel=0.5),

                           # Change brightness of images (50-150% of original value).
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),

                           # Improve or worsen the contrast of images.
                           iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                           # Convert each image to grayscale and then overlay the
                           # result with the original with random alpha. I.e. remove
                           # colors with varying strengths.
                           iaa.Grayscale(alpha=(0.0, 1.0)),

                           # In some images move pixels locally around (with random
                           # strengths).
                           sometimes(
                               iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                           ),

                           # In some images distort local areas with varying strength.
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                       ],
                       # do all of the above augmentations in random order
                       random_order=True
                       )
        ],
        # do all of the above augmentations in random order
        random_order=True
    )

    # images_aug = seq(images=images)

    return seq

def run_seq2():

    # if not isinstance(images,list):
    #     images=[images]
    # sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        # iaa.Fliplr(0.5),  # horizontal flips
        # iaa.Crop(percent=(0, 0.1)),  # random crops
        # iaa.Crop(percent=(0, 0.05)),
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Sometimes(0.5,
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-5, 5),
            # rotate=(-25, 25),
            shear=(-5, 5),
            order=[0, 1],
            cval=144,  # 填充像素值
        ))

    ], random_order=True)  # apply augmenters in random order

    # images_aug = seq(images=images)

    return seq


def simple_agu(image,target,seed=100,advanced=False,shape=(),last_class_id=True):
    """
    :param image:PIL image
    :param labels: [[x1,y1,x2,y2,class_id],[]]
    :return: image:PIL image
    """
    ia.seed(seed)
    image=np.asarray(image)
    labels=[[*box,label] for box,label in zip(target["boxes"].numpy(),target["labels"].numpy())]

    temp=[BoundingBox(*item[:-1],label=item[-1]) for item in labels]
    bbs=BoundingBoxesOnImage(temp,shape=image.shape)

    seq=run_seq() if advanced else run_seq2()
    # seq=run_seq2()

    # Augment BBs and images.
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()  # 处理图像外的边界框
    if shape:
        image_aug=ia.imresize_single_image(image_aug,shape)
        bbs_aug=bbs_aug.on(image_aug)

    # print coordinates before/after augmentation (see below)
    # use .x1_int, .y_int, ... to get integer coordinates
    # for i in range(len(bbs.bounding_boxes)):
    #         before = bbs.bounding_boxes[i]
    #         after = bbs_aug.bounding_boxes[i]
    #         print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
    #             i,
    #             before.x1, before.y1, before.x2, before.y2,
    #             after.x1, after.y1, after.x2, after.y2)
    #         )

    # image with BBs before/after augmentation (shown below)
    """
    image_before = bbs.draw_on_image(image, size=2)
    image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])

    skimage.io.imshow(image_before)
    skimage.io.show()

    skimage.io.imshow(image_after)
    skimage.io.show()
    # """

    # return image_aug, [[item.x1, item.y1, item.x2, item.y2, item.label] for item in bbs_aug.bounding_boxes] if last_class_id \
    #     else [[item.label,item.x1, item.y1, item.x2, item.y2] for item in bbs_aug.bounding_boxes]

    image_aug=PIL.Image.fromarray(image_aug)

    box=[]
    label=[]
    for item in bbs_aug.bounding_boxes:
        box.append([item.x1, item.y1, item.x2, item.y2])
        label.append(item.label)

    target["boxes"]=torch.as_tensor(box,dtype=torch.float32)
    target["labels"]=torch.as_tensor(label,dtype=torch.long)

    return image_aug,target


def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def clip_box(bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image

    """
    ar_ = (bbox_area(bbox))
    if (ar_>0).all():
        x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
        y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
        x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
        y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

        bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))

        delta_area = ((ar_ - bbox_area(bbox)) / ar_)

        mask = (delta_area < (1 - alpha)).astype(int)

        bbox = bbox[mask == 1, :]
    else:
        bbox = []

    return bbox


def rotate_im(image, angle):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black.

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    #    image = cv2.resize(image, (w,h))
    return image


def get_corners(bboxes):
    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def rotate_box(corners, angle, cx, cy, h, w):
    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int
        height of the image

    w : int
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """
    # print(corners)
    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    Returns
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final

def BGR2RGB( img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def BGR2HSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def HSV2BGR(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def glob_format(path,base_name = False):
    #print('--------pid:%d start--------------' % (os.getpid()))
    fmt_list = ('.jpg', '.jpeg', '.png',".xml")
    fs = []
    if not os.path.exists(path):return fs
    for root, dirs, files in os.walk(path):
        for file in files:
            item = os.path.join(root, file)
            # item = unicode(item, encoding='utf8')
            fmt = os.path.splitext(item)[-1]
            if fmt.lower() not in fmt_list:
                # os.remove(item)
                continue
            if base_name:fs.append(file)  # fs.append(os.path.splitext(file)[0])
            else:fs.append(item)
    #print('--------pid:%d end--------------' % (os.getpid()))
    return fs

def resize_boxes(boxes, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def resize_keypoints(keypoints, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_height, ratio_width = ratios

    keypoints[...,0] *= ratio_width  # x
    keypoints[...,1] *= ratio_height # y

    return keypoints


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets

def mosaic_resize(self,idx):
    # 做马赛克数据增强，详情参考：yolov4
    # 做resize
    index = torch.randperm(self.len).tolist()
    if idx + 3 >= self.len:
        idx = 0

    idx2 = index[idx + 1]
    idx3 = index[idx + 2]
    idx4 = index[idx + 3]

    img,mask, boxes, labels,img_path = self.load(idx)
    img2,mask2, boxes2, labels2,_ = self.load(idx2)
    img3,mask3, boxes3, labels3,_ = self.load(idx3)
    img4,mask4, boxes4, labels4,_ = self.load(idx4)

    h1, w1, _ = img.shape
    h2, w2, _ = img2.shape
    h3, w3, _ = img3.shape
    h4, w4, _ = img4.shape

    # img 取左上角,img2 右上角,img3 左下角,img4 右下角合成一张新图
    h = min((h1, h2, h3, h4))
    w = min((w1, w2, w3, h4))
    # h = max((h1, h2, h3, h4))//2
    # w = max((w1, w2, w3, h4))//2

    temp_img = np.zeros((2 * h, 2 * w, 3), np.uint8)
    if mask is not None:temp_masks = np.zeros((2 * h, 2 * w), np.uint8)
    temp_boxes = []
    temp_labels = []
    temp_img[0:h, 0:w] = cv2.resize(img, (w, h), interpolation=cv2.INTER_BITS)
    if mask is not None:temp_masks[0:h, 0:w] = cv2.resize(mask, (w, h), interpolation=cv2.INTER_BITS)
    temp_boxes.extend(resize_boxes(boxes, (h1, w1), (h, w)))
    temp_labels.extend(labels)

    temp_img[0:h, w:] = cv2.resize(img2, (w, h), interpolation=cv2.INTER_BITS)
    if mask is not None:temp_masks[0:h, w:] = cv2.resize(mask2, (w, h), interpolation=cv2.INTER_BITS)
    temp_boxes.extend(resize_boxes(boxes2, (h2, w2), (h, w)).add_(torch.tensor([w, 0, w, 0]).unsqueeze(0)))
    temp_labels.extend(labels2)

    temp_img[h:, 0:w] = cv2.resize(img3, (w, h), interpolation=cv2.INTER_BITS)
    if mask is not None:temp_masks[h:, 0:w] = cv2.resize(mask3, (w, h), interpolation=cv2.INTER_BITS)
    temp_boxes.extend(resize_boxes(boxes3, (h3, w3), (h, w)).add_(torch.tensor([0, h, 0, h]).unsqueeze(0)))
    temp_labels.extend(labels3)

    temp_img[h:, w:] = cv2.resize(img4, (w, h), interpolation=cv2.INTER_BITS)
    if mask is not None:temp_masks[h:, w:] = cv2.resize(mask4, (w, h), interpolation=cv2.INTER_BITS)
    temp_boxes.extend(resize_boxes(boxes4, (h4, w4), (h, w)).add_(torch.tensor([w, h, w, h]).unsqueeze(0)))
    temp_labels.extend(labels4)

    img = temp_img
    boxes = torch.stack(temp_boxes, 0).float()
    labels = torch.as_tensor(temp_labels, dtype=torch.long)
    if mask is not None:mask = temp_masks

    return img,mask,boxes,labels,img_path

def mosaic_crop(self,idx):
    # 做马赛克数据增强，详情参考：yolov4
    # 做裁剪
    min_pixes = 20
    index = torch.randperm(self.len).tolist()
    if idx + 3 >= self.len:
        idx = 0

    idx2 = index[idx + 1]
    idx3 = index[idx + 2]
    idx4 = index[idx + 3]

    img, mask, boxes, labels, img_path = self.load(idx)
    img2, mask2, boxes2, labels2, _ = self.load(idx2)
    img3, mask3, boxes3, labels3, _ = self.load(idx3)
    img4, mask4, boxes4, labels4, _ = self.load(idx4)

    h1, w1, _ = img.shape
    h2, w2, _ = img2.shape
    h3, w3, _ = img3.shape
    h4, w4, _ = img4.shape

    # img 取左上角,img2 右上角,img3 左下角,img4 右下角合成一张新图
    # h = min((h1, h2, h3, h4))
    # w = min((w1, w2, w3, h4))
    h = max((h1, h2, h3, h4))
    w = max((w1, w2, w3, h4))

    xc = int(random.uniform(w * 0.5, w * 1.5)) # mosaic center x, y
    yc = int(random.uniform(h * 0.5, h * 1.5))
    # xc = int(random.uniform(w * 0.3, w * 0.8))  # mosaic center x, y
    # yc = int(random.uniform(h * 0.3, h * 0.8))

    # temp_img = np.zeros((2 * h, 2 * w, 3), np.uint8)
    temp_labels = []
    temp_boxes = []

    dy1 = min(yc,h1,h2)
    dx1 = min(xc,w1,w3)

    dy2 = min(dy1+h3,dy1+h4,2*h)
    dx2 = min(dx1+w2,dx1+w4,2*w)
    # dy2 = min(dy1 + h3, dy1 + h4, h)
    # dx2 = min(dx1 + w2, dx1 + w4, w)

    temp_img = np.zeros((dy2, dx2, 3), np.uint8)
    if mask is not None:temp_masks = np.zeros((dy2, dx2), np.uint8)

    temp_img[0:dy1,0:dx1] = img[0:dy1,0:dx1]
    if mask is not None:temp_masks[0:dy1,0:dx1] = mask[0:dy1,0:dx1]
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, dx1)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, dy1)
    keep = (boxes[..., 2:] - boxes[..., :2] > min_pixes).sum(-1) == 2
    boxes = boxes[keep]
    labels = labels[keep]
    temp_boxes.extend(boxes.add_(torch.tensor([0, 0, 0, 0]).unsqueeze(0)))
    temp_labels.extend(labels)

    temp_img[0:dy1, dx1:dx2] = img2[0:dy1, 0:dx2-dx1]
    if mask is not None: temp_masks[0:dy1, dx1:dx2] = mask2[0:dy1, 0:dx2-dx1]
    boxes2[..., [0, 2]] = boxes2[..., [0, 2]].clamp(0, dx2-dx1)
    boxes2[..., [1, 3]] = boxes2[..., [1, 3]].clamp(0, dy1)
    keep = (boxes2[..., 2:] - boxes2[..., :2] > min_pixes).sum(-1) == 2
    boxes2 = boxes2[keep]
    labels2 = labels2[keep]
    temp_boxes.extend(boxes2.add_(torch.tensor([dx1, 0, dx1, 0]).unsqueeze(0)))
    temp_labels.extend(labels2)

    temp_img[dy1:dy2, 0:dx1] = img3[0:dy2-dy1, 0:dx1]
    if mask is not None: temp_masks[dy1:dy2, 0:dx1] = mask3[0:dy2-dy1, 0:dx1]
    boxes3[..., [0, 2]] = boxes3[..., [0, 2]].clamp(0, dx1)
    boxes3[..., [1, 3]] = boxes3[..., [1, 3]].clamp(0, dy2-dy1)
    keep = (boxes3[..., 2:] - boxes3[..., :2] > min_pixes).sum(-1) == 2
    boxes3 = boxes3[keep]
    labels3 = labels3[keep]
    temp_boxes.extend(boxes3.add_(torch.tensor([0, dy1, 0, dy1]).unsqueeze(0)))
    temp_labels.extend(labels3)

    temp_img[dy1:dy2, dx1:dx2] = img4[0:dy2 - dy1, 0:dx2-dx1]
    if mask is not None: temp_masks[dy1:dy2, dx1:dx2] = mask4[0:dy2 - dy1, 0:dx2-dx1]
    boxes4[..., [0, 2]] = boxes4[..., [0, 2]].clamp(0, dx2-dx1)
    boxes4[..., [1, 3]] = boxes4[..., [1, 3]].clamp(0, dy2 - dy1)
    keep = (boxes4[..., 2:] - boxes4[..., :2] > min_pixes).sum(-1) == 2
    boxes4 = boxes4[keep]
    labels4 = labels4[keep]
    temp_boxes.extend(boxes4.add_(torch.tensor([dx1, dy1, dx1, dy1]).unsqueeze(0)))
    temp_labels.extend(labels4)

    if len(temp_labels)>0:
        img = temp_img
        boxes = torch.stack(temp_boxes, 0).float()
        labels = torch.as_tensor(temp_labels, dtype=torch.long)
        if mask is not None:mask = temp_masks
    else:
        img, mask, boxes, labels, img_path = self.load(idx)

    return img, mask, boxes, labels, img_path

def mosaic_origin(self,index):
    self.hyp={
        'degrees': 1.98 * 0,  # image rotation (+/- deg)
        'translate': 0.05 * 0,  # image translation (+/- fraction)
        'scale': 0.05 * 0,  # image scale (+/- gain)
        'shear': 0.641 * 0}  # image shear (+/- deg)

    img, mask, _, _, img_path = self.load(index)

    labels4 = []
    s = max(img.shape[:2])
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    indices = [index] + [random.randint(0, self.len - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        img,mask, boxes, labels,_ = self.load(index)

        h, w = img.shape[:2]
        labels = torch.cat((labels.float().unsqueeze(-1),boxes),-1)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            if mask is not None: temp_masks = np.zeros((s * 2, s * 2), np.uint8)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        if mask is not None:temp_masks[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b
        if len(labels)>0:
            labels = labels+torch.tensor([0,padw,padh,padw,padh]).unsqueeze(0)
        labels4.append(labels.numpy())

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=self.hyp['degrees'],
                                  translate=self.hyp['translate'],
                                  scale=self.hyp['scale'],
                                  shear=self.hyp['shear'],
                                  border=-s // 2)  # border to remove


    boxes = torch.tensor(labels4[:,1:]).float()
    labels = torch.as_tensor(labels4[:,0], dtype=torch.long)
    if mask is not None:mask=temp_masks
    return img4,mask,boxes,labels,img_path

def mixup(self,idx):
    index = torch.randperm(self.len).tolist()
    if idx + 1 >= self.len:
        idx = 0

    idx2 = index[idx + 1]
    img, mask, boxes, labels,img_path = self.load(idx)
    img2, mask2, boxes2, labels2,_ = self.load(idx2)

    h1, w1, _ = img.shape
    h2, w2, _ = img2.shape

    h = max((h1, h2))
    w = max((w1, w2))

    temp_img1 = np.zeros((h, w, 3), np.uint8)
    temp_img2 = np.zeros((h, w, 3), np.uint8)
    temp_img1[:h1,:w1] = img
    temp_img2[:h2,:w2] = img2

    if mask is not None:
        temp_mask1 = np.zeros((h, w), np.uint8)
        temp_mask2 = np.zeros((h, w), np.uint8)
        temp_mask1[:h1, :w1] = mask
        temp_mask2[:h2, :w2] = mask2

    gamma = np.random.uniform(0.4,0.6)
    if mask is not None:temp_mask = np.clip(cv2.addWeighted(temp_mask1, gamma, temp_mask2, 1-gamma, 0.0), 0, 255).astype(np.uint8)
    temp_img = np.clip(cv2.addWeighted(temp_img1,gamma,temp_img2,1-gamma,0.0),0,255).astype(np.uint8)

    temp_boxes = []
    temp_labels = []
    temp_boxes.extend(boxes)
    temp_boxes.extend(boxes2)
    temp_labels.extend(labels)
    temp_labels.extend(labels2)

    img = temp_img
    boxes = torch.stack(temp_boxes, 0).float()
    labels = torch.as_tensor(temp_labels, dtype=torch.long)
    if mask is not None:mask=temp_mask
    return img,mask,boxes, labels,img_path


def flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data
