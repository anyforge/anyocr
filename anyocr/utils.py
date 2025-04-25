from typing import Any, Tuple,Dict, List, Optional, Union
import cv2
import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image, UnidentifiedImageError



def reduce_max_side(
    img: np.ndarray, max_side_len: int = 2000
) -> Tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]

    ratio = 1.0
    if max(h, w) > max_side_len:
        if h > w:
            ratio = float(max_side_len) / h
        else:
            ratio = float(max_side_len) / w

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = int(round(resize_h / 32) * 32)
    resize_w = int(round(resize_w / 32) * 32)

    try:
        if int(resize_w) <= 0 or int(resize_h) <= 0:
            raise ResizeImgError("resize_w or resize_h is less than or equal to 0")
        img = cv2.resize(img, (resize_w, resize_h))
    except Exception as exc:
        raise ResizeImgError() from exc

    ratio_h = h / resize_h
    ratio_w = w / resize_w
    return img, ratio_h, ratio_w


def increase_min_side(
    img: np.ndarray, min_side_len: int = 30
) -> Tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]

    ratio = 1.0
    if min(h, w) < min_side_len:
        if h < w:
            ratio = float(min_side_len) / h
        else:
            ratio = float(min_side_len) / w

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = int(round(resize_h / 32) * 32)
    resize_w = int(round(resize_w / 32) * 32)

    try:
        if int(resize_w) <= 0 or int(resize_h) <= 0:
            raise ResizeImgError("resize_w or resize_h is less than or equal to 0")
        img = cv2.resize(img, (resize_w, resize_h))
    except Exception as exc:
        raise ResizeImgError() from exc

    ratio_h = h / resize_h
    ratio_w = w / resize_w
    return img, ratio_h, ratio_w


def add_round_letterbox(
    img: np.ndarray,
    padding_tuple: Tuple[int, int, int, int],
) -> np.ndarray:
    padded_img = cv2.copyMakeBorder(
        img,
        padding_tuple[0],
        padding_tuple[1],
        padding_tuple[2],
        padding_tuple[3],
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return padded_img


class ResizeImgError(Exception):
    pass


InputType = Union[str, np.ndarray, bytes, Path, Image.Image]
class LoadImage:
    def __init__(self):
        pass

    def __call__(self, img: InputType) -> np.ndarray:
        if not isinstance(img, InputType.__args__):
            raise LoadImageError(
                f"The img type {type(img)} does not in {InputType.__args__}"
            )

        origin_img_type = type(img)
        img = self.load_img(img)
        img = self.convert_img(img, origin_img_type)
        return img

    def load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            self.verify_exist(img)
            try:
                img = self.img_to_ndarray(Image.open(img))
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = self.img_to_ndarray(Image.open(BytesIO(img)))
            return img

        if isinstance(img, np.ndarray):
            return img

        if isinstance(img, Image.Image):
            return self.img_to_ndarray(img)

        raise LoadImageError(f"{type(img)} is not supported!")

    def img_to_ndarray(self, img: Image.Image) -> np.ndarray:
        if img.mode == "1":
            img = img.convert("L")
            return np.array(img)
        return np.array(img)

    def convert_img(self, img: np.ndarray, origin_img_type: Any) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3:
            channel = img.shape[2]
            if channel == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if channel == 2:
                return self.cvt_two_to_three(img)

            if channel == 3:
                if issubclass(origin_img_type, (str, Path, bytes, Image.Image)):
                    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img

            if channel == 4:
                return self.cvt_four_to_three(img)

            raise LoadImageError(
                f"The channel({channel}) of the img is not in [1, 2, 3, 4]"
            )

        raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")

    @staticmethod
    def cvt_two_to_three(img: np.ndarray) -> np.ndarray:
        """gray + alpha → BGR"""
        img_gray = img[..., 0]
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        img_alpha = img[..., 1]
        not_a = cv2.bitwise_not(img_alpha)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(img_bgr, img_bgr, mask=img_alpha)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """RGBA → BGR"""
        r, g, b, a = cv2.split(img)
        new_img = cv2.merge((b, g, r))

        not_a = cv2.bitwise_not(a)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(new_img, new_img, mask=a)

        mean_color = np.mean(new_img)
        if mean_color <= 0.0:
            new_img = cv2.add(new_img, not_a)
        else:
            new_img = cv2.bitwise_not(new_img)
        return new_img

    @staticmethod
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")


class LoadImageError(Exception):
    pass