import cv2
import numpy as np


def resize_crop(image, size=224):
    min_dim = np.argmin(image.shape[:2])
    scale_percent = size / image.shape[min_dim]
    width = round(image.shape[1] * scale_percent)
    height = round(image.shape[0] * scale_percent)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if min_dim == 1:
        image = image[:size]
    else:
        image = image[:, :size]
    return image


def resize_pad(image, handsize=200, size=224, random=True):
    max_dim = np.argmax(image.shape[:2])
    scale_percent = handsize / image.shape[max_dim]
    width = round(image.shape[1] * scale_percent)
    height = round(image.shape[0] * scale_percent)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    padd1 = np.random.randint(0, size - image.shape[1] + 1)
    if not random:
        padd1 = (size - image.shape[1]) // 2
    padd0 = np.random.randint(0, size - image.shape[0] + 1)
    if not random:
        padd1 = (size - image.shape[0]) // 2
    image = cv2.copyMakeBorder(
        image,
        padd0,
        size - image.shape[0] - padd0,
        padd1,
        size - image.shape[1] - padd1,
        cv2.BORDER_CONSTANT,
        0,
    )
    return image


def normalize(imgs):
    return swapaxis(imgs).astype(np.float32) / 255


def swapaxis(imgs):
    return np.moveaxis(imgs, -1, 1)


def crop_hand_cnn(img, hands, margin=0.07, random=False):
    # plays recording from camera and processes each image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    cropped_results = []
    if not results.multi_hand_landmarks:
        return None, None
    for handLms in results.multi_hand_landmarks:
        landmark_listx = []
        landmark_listy = []
        for lm in handLms.landmark:
            h, w, c = img.shape
            # here we multiply by the width and height because the landmarks are auto-normalized based on
            # the width and height of the displayed image
            landmark_listx.append((lm.x * w))
            landmark_listy.append((lm.y * h))
        end = (
            int(max(landmark_listx)) + int(margin * w),
            int(max(landmark_listy)) + int(margin * h),
        )
        start = (
            max(0, int(min(landmark_listx)) - int(margin * w)),
            max(0, int(min(landmark_listy)) - int(margin * h)),
        )
        cropped_img = img[start[1] : end[1], start[0] : end[0]]
        if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
            continue
        # print(cropped_img.shape)
        cropped_results.append(resize_pad(cropped_img, random=True))

    joints = np.array(
        [
            [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]
            for hand_lms in results.multi_hand_landmarks
        ]
    )

    if len(cropped_results) == 0:
        return None, None
    return np.array(cropped_results, dtype=np.float32), joints[None, ...].astype(
        np.float32
    )


def normalize_joints(total_x):
    total_x = total_x - total_x[:, :1, :]
    factor = np.mean(
        np.linalg.norm(total_x - total_x[:, 0:1, :], axis=-1, keepdims=True),
        axis=-2,
        keepdims=True,
    )
    if (factor != 0).all():
        total_x /= factor
    return normalize_rotation(to_polar(total_x))


def to_polar(total_x):
    angles = np.arctan2(total_x[:, :, 1], total_x[:, :, 0])
    radi = np.linalg.norm(total_x[:, :, :], axis=-1)
    depth = total_x[:, :, 2]
    # (N, 21)
    return np.concatenate(
        [angles[..., None], radi[..., None], depth[..., None]], axis=-1
    )


def normalize_rotation(total_x):
    shift = total_x[:, 8, 0] - 0
    total_x[:, :, 0] -= shift[..., None]
    return total_x


def random_scale(total_x):
    rand_scale = np.random.uniform(0.5, 2)
    total_x[:, :, 1] *= rand_scale
    total_x[:, :, 2] *= rand_scale
    return total_x
