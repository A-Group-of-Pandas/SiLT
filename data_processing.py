import cv2
import numpy as np


# DATA PROCESSING
def read_data(data_name: str, label_name: str):
    # reads in data from the dataset
    data = np.load(f"data/{data_name}.npy")
    labels = np.load(f"data/{label_name}.npy")
    # separates the data into training data/labels and returns them
    return data, labels


def read_batch(batch_size: int = 32, train: bool = True, joints: bool = True):
    if joints:
        if train:
            total_x, total_y = read_data("train_joints", "train_lab_joint")
        else:
            total_x, total_y = read_data("test_joints", "test_lab_joint")
    else:
        if train:
            total_x, total_y = read_data("images", "labels")
        else:
            total_x, total_y = read_data("test_images", "test_labels")
    indices = np.arange(len(total_y))
    np.random.shuffle(indices)
    for batch_i in range(len(total_y) // batch_size):
        idx = indices[batch_i * batch_size : (batch_i + 1) * batch_size]
        if joints:
            yield normalize_joints(total_x[idx]), total_y[idx]
        else:
            # imgs = [cv2.copyMakeBorder(img, 0,224-img.shape[0],0,224-img.shape[1],cv2.BORDER_CONSTANT,0) for img in total_x[idx]]
            yield normalize(total_x[idx]), total_y[idx]
    # yields a generator for batches of data


# ====================================================================================
# IMAGE PROCESSING


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


def resize_pad(image, handsize=200, size=224):
    max_dim = np.argmax(image.shape[:2])
    scale_percent = handsize / image.shape[max_dim]
    width = round(image.shape[1] * scale_percent)
    height = round(image.shape[0] * scale_percent)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    padd1 = (size - image.shape[1] + 1) // 2
    padd0 = (size - image.shape[0] + 1) // 2
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


# ====================================================================================

# j = np.random.random((1,1,21,3))


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


def random_rotate(total_x):
    angle = np.random.uniform(-1, 1) * np.pi / 4
    total_x[:, :, 0] += angle
    return total_x


def random_scale(total_x):
    rand_scale = np.random.uniform(0.2, 5)
    total_x[:, :, 1] *= rand_scale
    total_x[:, :, 2] *= rand_scale
    return total_x


def random_flip(total_x):
    if np.random.randint(0, 2) == 0:
        total_x[:, :, 0] = np.where(
            total_x[:, :, 0] > 0, np.pi - total_x[:, :, 0], -np.pi - total_x[:, :, 0]
        )
        return total_x
    else:
        return total_x


def data_augmentation(total_x):
    # (N,1,21,3)
    total_x = random_scale(total_x)
    total_x = random_flip(total_x)
    return random_rotate(total_x)


# print(normalize_joints(j))
