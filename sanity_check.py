import torch
import torchvision.transforms as transforms
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

import onnxruntime

# Load the SAM model (you may need to adjust the model loading based on the specific implementation)
def load_sam_model(model_path):
    # Instantiate the model registry and predictor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry["vit_b"](checkpoint=model_path).to(device)
    return sam

# Define preprocessing transformation
preprocess = transforms.Compose([
    #transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375]),
])

def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> list[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer

def get_image_embeddings(image_path, model):
    # Load and preprocess the image
    image = Image.open(image_path)#.convert("RGB")
    image = image.resize((1024, 1024))
    input_tensor = np.zeros((1, 3, 1024, 1024))#preprocess(image)
    image_arr = np.array(image)
    height, width, _ = image_arr.shape
    for y in range(height):
        for x in range(width):
            pixel = image_arr[y, x]
            r, g, b = pixel
            input_tensor[0, 0, y, x] = (r - 123.675) / 58.395
            input_tensor[0, 1, y, x] = (g - 116.28) / 57.120
            input_tensor[0, 2, y, x] = (b - 103.53) / 57.375
    np.savetxt("./images/preprocesspy.txt", input_tensor.flatten(), delimiter=',')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_batch = torch.from_numpy(input_tensor).to(torch.float32).to(device)

    # Run the image through the model
    with torch.no_grad():
        # If the SAM model requires specific input format, adjust here
        features = model(input_batch)

    # Extract embeddings (features) from the model
    return features.cpu().numpy()

def _generate_masks(image: np.ndarray):
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, 0, 0.341
        )

        print(f"crop_boxes: {crop_boxes}")
        print(f"layer_idxs: {layer_idxs}")

        # Iterate over image crops
        # data = MaskData()
        # for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
        #     crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
        #     data.cat(crop_data)

        # # Remove duplicate masks between crops
        # if len(crop_boxes) > 1:
        #     # Prefer masks from smaller crops
        #     scores = 1 / box_area(data["crop_boxes"])
        #     scores = scores.to(data["boxes"].device)
        #     keep_by_nms = batched_nms(
        #         data["boxes"].float(),
        #         scores,
        #         torch.zeros_like(data["boxes"][:, 0]),  # categories
        #         iou_threshold=self.crop_nms_thresh,
        #     )
        #     data.filter(keep_by_nms)

        # data.to_numpy()
        # return data

def generate_crop_boxes(
    im_size: tuple[int, ...], n_layers: int, overlap_ratio: float
) -> tuple[list[list[int]], list[int]]:
    """
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    """
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # Original image
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs

# Example usage
model_path = '/tmp/sam_vit_b_01ec64.pth'  # Path to your SAM model
image_path = './images/deer.jpg'
image = Image.open(image_path)
image_arr = np.array(image)
_generate_masks(image_arr)

# sam_model = load_sam_model(model_path)
# embeddings = get_image_embeddings(image_path, sam_model.image_encoder)
# flat = embeddings.flatten()
# np.savetxt("./images/embeddings.txt", flat, delimiter=',')
#print(embeddings.shape)  # Output shape will depend on the model's output


# ort_session = onnxruntime.InferenceSession("./models/vit_b_decoder.onnx")
# predictor = SamPredictor(sam_model)
# image = Image.open('./images/deer.jpg')
# predictor.set_image(image)