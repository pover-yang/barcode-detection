from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor

# from torchvision.utils import _log_api_usage_once


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.
    Args:
        backbone (nn.Module): extract image features
        rpn (nn.Module): generate proposals
        roi_heads (nn.Module): takes the features + the proposals and computes detections from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into the model
    """

    def __init__(self,
                 backbone: nn.Module,
                 rpn: nn.Module,
                 roi_heads: nn.Module,
                 transform: nn.Module) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self,
                images: List[Tensor],
                targets: Optional[List[Dict[str, Tensor]]] = None):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise RuntimeError("targets should not be none when in training mode")
        else:
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    assert len(boxes.shape) == 2 and boxes.shape[-1] == 4
                    # torch._assert(
                    #     len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                    #     f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                    # )
                else:
                    raise TypeError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes,
                                                original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, detections
