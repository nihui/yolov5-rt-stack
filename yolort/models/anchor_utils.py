# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
from typing import Tuple, List

import torch
from torch import nn, Tensor


class AnchorGenerator(nn.Module):
    def __init__(
        self,
        strides: List[int],
        anchor_grids: List[List[float]],
    ):
        super().__init__()
        assert len(strides) == len(anchor_grids)
        self.num_layers = len(anchor_grids)
        self.num_anchors = len(anchor_grids[0]) // 2
        self.register_buffer(
            "anchors", torch.tensor(anchor_grids).float().view(self.num_layers, -1, 2)
        )
        self.strides = strides

    def _generate_anchor_grids(
        self,
        grid_sizes: List[List[int]],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[List[Tensor], List[Tensor]]:
        grids = []
        anchor_grids = []

        for i, sizes in enumerate(grid_sizes):
            height, width = sizes

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            widths = torch.arange(0, width, dtype=torch.int32, device=device).to(dtype=dtype)
            heights = torch.arange(0, height, dtype=torch.int32, device=device).to(dtype=dtype)

            shift_y, shift_x = torch.meshgrid(heights, widths)

            grid = torch.stack((shift_x, shift_y), 2).expand(
                (1, self.num_anchors, height, width, 2)
            )
            anchor_grid = (
                (self.anchors[i].clone() * self.strides[i])
                .view((1, self.num_anchors, 1, 1, 2))
                .expand((1, self.num_anchors, height, width, 2))
            )

            grids.append(grid)
            anchor_grids.append(anchor_grid)
        return grids, anchor_grids

    def forward(self, feature_maps: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        grids, anchor_grids = self._generate_anchor_grids(grid_sizes, dtype, device)
        return grids, anchor_grids
