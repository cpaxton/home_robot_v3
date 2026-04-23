# Copyright (c) Hello Robot, Inc.
#
# Unit tests for emet.mapping.instance: Instance, InstanceView, InstanceMemory.
# These test the core data structures and association logic without simulation.

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Import instance module directly to avoid heavy emet.mapping.__init__ chain
_instance_path = Path(__file__).resolve().parents[2] / "emet" / "mapping" / "instance.py"
_spec = importlib.util.spec_from_file_location("_instance", str(_instance_path))
_mod = importlib.util.module_from_spec(_spec)
_mod.__package__ = "emet.mapping"
sys.modules["_instance"] = _mod
_spec.loader.exec_module(_mod)

Instance = _mod.Instance
InstanceMemory = _mod.InstanceMemory
InstanceView = _mod.InstanceView
_bbox3d_iou = _mod._bbox3d_iou


class TestInstanceView:
    def test_get_image_returns_uint8_hwc(self):
        crop = torch.rand(50, 40, 3)  # float [0,1]
        view = InstanceView(
            cropped_image=crop,
            mask=torch.ones(50, 40, 1),
            bbox_xyxy=torch.tensor([10, 20, 50, 70]),
            cam_to_world=torch.eye(4),
        )
        img = view.get_image()
        assert img.dtype == np.uint8
        assert img.shape == (50, 40, 3)

    def test_get_pose_returns_cam_to_world(self):
        pose = torch.eye(4)
        pose[0, 3] = 1.5
        view = InstanceView(
            cropped_image=torch.zeros(10, 10, 3),
            mask=torch.ones(10, 10),
            bbox_xyxy=torch.tensor([0, 0, 10, 10]),
            cam_to_world=pose,
        )
        assert torch.allclose(view.get_pose(), pose)


class TestInstance:
    def _make_instance(self):
        inst = Instance(instance_id=0, global_id=0, category_id=5, score=0.9)
        pts = torch.tensor([[1.0, 2.0, 0.5], [1.1, 2.1, 0.6], [0.9, 1.9, 0.4]])
        inst.point_cloud = pts
        inst.point_cloud_rgb = torch.rand(3, 3)
        inst.update_bounds()
        return inst

    def test_get_center(self):
        inst = self._make_instance()
        center = inst.get_center()
        assert center.shape == (3,)
        np.testing.assert_allclose(center.numpy(), [1.0, 2.0, 0.5], atol=0.15)

    def test_get_median(self):
        inst = self._make_instance()
        median = inst.get_median()
        assert median.shape == (3,)

    def test_get_closest_point(self):
        inst = self._make_instance()
        closest = inst.get_closest_point(np.array([1.0, 2.0, 0.5]))
        assert closest.shape == (3,)

    def test_bounds(self):
        inst = self._make_instance()
        assert inst.bounds is not None
        assert inst.bounds.shape == (3, 2)
        # min should be <= max
        assert (inst.bounds[:, 0] <= inst.bounds[:, 1]).all()

    def test_get_best_view(self):
        inst = Instance(instance_id=1, global_id=1)
        v1 = InstanceView(
            cropped_image=torch.zeros(5, 5, 3),
            mask=torch.ones(5, 5),
            bbox_xyxy=torch.zeros(4),
            cam_to_world=torch.eye(4),
            score=0.5,
            timestep=0,
        )
        v2 = InstanceView(
            cropped_image=torch.zeros(5, 5, 3),
            mask=torch.ones(5, 5),
            bbox_xyxy=torch.zeros(4),
            cam_to_world=torch.eye(4),
            score=0.9,
            timestep=1,
        )
        inst.add_view(v1)
        inst.add_view(v2)
        best = inst.get_best_view()
        assert best.score == 0.9

    def test_get_image_embedding_mean(self):
        inst = Instance(instance_id=2, global_id=2)
        emb1 = torch.randn(8)
        emb2 = torch.randn(8)
        v1 = InstanceView(
            cropped_image=torch.zeros(5, 5, 3),
            mask=torch.ones(5, 5),
            bbox_xyxy=torch.zeros(4),
            cam_to_world=torch.eye(4),
            embedding=emb1,
        )
        v2 = InstanceView(
            cropped_image=torch.zeros(5, 5, 3),
            mask=torch.ones(5, 5),
            bbox_xyxy=torch.zeros(4),
            cam_to_world=torch.eye(4),
            embedding=emb2,
        )
        inst.add_view(v1)
        inst.add_view(v2)
        agg = inst.get_image_embedding(aggregation_method="mean", normalize=True)
        assert agg.shape == (8,)
        assert abs(agg.norm().item() - 1.0) < 1e-5


class TestBBox3dIoU:
    def test_identical_boxes(self):
        b = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        assert abs(_bbox3d_iou(b, b) - 1.0) < 1e-5

    def test_no_overlap(self):
        a = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        b = torch.tensor([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]])
        assert _bbox3d_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = torch.tensor([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
        b = torch.tensor([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]])
        iou = _bbox3d_iou(a, b)
        assert 0 < iou < 1


class TestInstanceMemory:
    def _make_frame_data(self, n_instances=2, H=100, W=100):
        """Create synthetic instance segmentation data."""
        instance_seg = -torch.ones(H, W, dtype=torch.long)
        point_cloud = torch.rand(H, W, 3)
        image = (torch.rand(3, H, W) * 255).byte()
        cam_to_world = torch.eye(4)

        # Place two instances in different regions
        instance_seg[10:50, 10:50] = 0
        instance_seg[60:90, 60:90] = 1
        point_cloud[10:50, 10:50] = torch.tensor([1.0, 0.0, 0.5])
        point_cloud[60:90, 60:90] = torch.tensor([3.0, 0.0, 0.5])

        instance_classes = torch.tensor([5, 10], dtype=torch.long)
        instance_scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
        valid_points = torch.ones(H, W, dtype=torch.bool)

        return {
            "instance_seg": instance_seg,
            "point_cloud": point_cloud,
            "image": image,
            "cam_to_world": cam_to_world,
            "instance_classes": instance_classes,
            "instance_scores": instance_scores,
            "valid_points": valid_points,
        }

    def test_process_and_associate(self):
        mem = InstanceMemory(
            num_envs=1,
            min_pixels_for_instance_view=10,
            min_percent_for_instance_view=0.001,
            min_instance_thickness=0.0,
            min_instance_vol=0.0,
            max_instance_vol=100.0,
            min_instance_height=0.0,
            max_instance_height=10.0,
        )
        data = self._make_frame_data()
        mem.process_instances_for_env(
            env_id=0,
            instance_seg=data["instance_seg"],
            point_cloud=data["point_cloud"],
            image=data["image"],
            cam_to_world=data["cam_to_world"],
            instance_classes=data["instance_classes"],
            instance_scores=data["instance_scores"],
            background_instance_labels=[-1],
            valid_points=data["valid_points"],
        )
        mem.associate_instances_to_memory()

        instances = mem.get_all_instances(env_id=0)
        assert len(instances) == 2
        for inst in instances:
            assert inst.point_cloud is not None
            assert inst.bounds is not None
            assert len(inst.instance_views) == 1

    def test_association_merges_same_object(self):
        """Two frames of the same object (overlapping bbox) should merge."""
        mem = InstanceMemory(
            num_envs=1,
            min_pixels_for_instance_view=10,
            min_percent_for_instance_view=0.001,
            min_instance_thickness=0.0,
            min_instance_vol=0.0,
            max_instance_vol=100.0,
            min_instance_height=0.0,
            max_instance_height=10.0,
        )
        # Use spread-out points so the bounding box has nonzero volume
        H, W = 100, 100
        instance_seg = -torch.ones(H, W, dtype=torch.long)
        instance_seg[10:50, 10:50] = 0
        point_cloud = torch.rand(H, W, 3) * 0.5
        point_cloud[10:50, 10:50] = torch.rand(40, 40, 3) * 0.5 + torch.tensor([1.0, 0.0, 0.3])
        image = (torch.rand(3, H, W) * 255).byte()
        cam_to_world = torch.eye(4)
        instance_classes = torch.tensor([5], dtype=torch.long)
        instance_scores = torch.tensor([0.9], dtype=torch.float32)
        valid_points = torch.ones(H, W, dtype=torch.bool)

        # First frame
        mem.process_instances_for_env(
            env_id=0, instance_seg=instance_seg, point_cloud=point_cloud,
            image=image, cam_to_world=cam_to_world,
            instance_classes=instance_classes, instance_scores=instance_scores,
            background_instance_labels=[-1], valid_points=valid_points,
        )
        mem.associate_instances_to_memory()

        # Second frame with same object (same bbox region, slightly shifted points)
        point_cloud2 = point_cloud.clone()
        point_cloud2[10:50, 10:50] += torch.rand(40, 40, 3) * 0.05
        mem.process_instances_for_env(
            env_id=0, instance_seg=instance_seg, point_cloud=point_cloud2,
            image=image, cam_to_world=cam_to_world,
            instance_classes=instance_classes, instance_scores=instance_scores,
            background_instance_labels=[-1], valid_points=valid_points,
        )
        mem.associate_instances_to_memory()

        instances = mem.get_all_instances(env_id=0)
        merged = [i for i in instances if len(i.instance_views) >= 2]
        assert len(merged) >= 1, "Same object from two frames should be merged"

    def test_pop_global_instance(self):
        mem = InstanceMemory(
            num_envs=1,
            min_pixels_for_instance_view=10,
            min_percent_for_instance_view=0.001,
            min_instance_thickness=0.0,
            min_instance_vol=0.0,
            max_instance_vol=100.0,
            min_instance_height=0.0,
            max_instance_height=10.0,
        )
        data = self._make_frame_data()
        mem.process_instances_for_env(
            env_id=0,
            instance_seg=data["instance_seg"],
            point_cloud=data["point_cloud"],
            image=data["image"],
            cam_to_world=data["cam_to_world"],
            instance_classes=data["instance_classes"],
            instance_scores=data["instance_scores"],
            background_instance_labels=[-1],
            valid_points=data["valid_points"],
        )
        mem.associate_instances_to_memory()

        instances = mem.get_all_instances(env_id=0)
        assert len(instances) == 2
        gid = instances[0].global_id
        removed = mem.pop_global_instance(env_id=0, global_instance_id=gid)
        assert removed is not None
        assert len(mem.get_all_instances(env_id=0)) == 1
