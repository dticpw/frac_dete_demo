from __future__ import annotations

from pathlib import Path


TARGET = Path("D:/python/anaconda/envs/fracmed/Lib/site-packages/nnunetv2/imageio/natural_image_reader_writer.py")


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    original = text
    text = text.replace("from skimage import io\n", "")
    text = text.replace(
        "    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:\n"
        "        images = []\n"
        "        for f in image_fnames:\n"
        "            npy_img = io.imread(f)\n",
        "    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:\n"
        "        import imageio.v3 as iio\n"
        "\n"
        "        images = []\n"
        "        for f in image_fnames:\n"
        "            npy_img = iio.imread(f)\n",
    )
    text = text.replace(
        "    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:\n"
        "        io.imsave(output_fname, seg[0].astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False), check_contrast=False)\n",
        "    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:\n"
        "        import imageio.v3 as iio\n"
        "\n"
        "        iio.imwrite(output_fname, seg[0].astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False))\n",
    )
    if text == original:
        print("No changes needed; patch already applied or target format changed.")
        return
    TARGET.write_text(text, encoding="utf-8")
    print(f"Patched {TARGET}")


if __name__ == "__main__":
    main()
