from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel
import torch

def main():
    # Load sam model from downloaded pth file
    sam = sam_model_registry["vit_b"](checkpoint="/tmp/sam_vit_b_01ec64.pth")

    model_output_directory = "./models/"

    # Export images encoder from SAM model to ONNX
    with open(model_output_directory + "vit_b_encoder.onnx", "wb+") as f:
        torch.onnx.export(
            f=f,
            model=sam.image_encoder,
            args=torch.randn(1, 3, 1024, 1024),
            input_names=["images"],
            output_names=["embeddings"],
            export_params=True
        )

    # Export mask decoder from SAM model to ONNX
    onnx_model = SamOnnxModel(sam, return_single_mask=False)
    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]
    with open(model_output_directory + "vit_b_decoder.onnx", "wb+") as f:
        torch.onnx.export(
            f=f,
            model=onnx_model,
            args=tuple(dummy_inputs.values()),
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes={
                "point_coords": {1: "num_points"},
                "point_labels": {1: "num_points"}
            },
            export_params=True,
            opset_version=17,
            do_constant_folding=True
        )

if __name__ == "__main__":
    main()