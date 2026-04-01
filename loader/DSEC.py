import os
from pathlib import Path

# 导入 hdf5plugin 以自动注册 HDF5 压缩插件（DSEC 数据集需要）
try:
    import hdf5plugin
except ImportError:
    print("Warning: hdf5plugin not installed. DSEC data loading may fail.")
    print("Install with: pip install hdf5plugin")

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


def load_flow_png(flow_path, flow_scale=1.0):
    """读取 DSEC 双通道光流 PNG，并按 flow_scale 进行缩放。"""
    flow_img = cv2.imread(str(flow_path), cv2.IMREAD_UNCHANGED)
    if flow_img is None:
        return torch.zeros(2, 480, 640)
    flow_u = (flow_img[:, :, 0].astype(np.float32) - 32768) / 128.0
    flow_v = (flow_img[:, :, 1].astype(np.float32) - 32768) / 128.0
    flow = np.stack([flow_u, flow_v], axis=0)
    if flow_scale not in (None, 0, 1.0):
        flow = flow / float(flow_scale)
    return torch.from_numpy(flow)


class DSECEventFlow(Dataset):
    """
    生成与 HREMEventFlow 兼容的样本格式：
      - event_volume_old: [C,H,W]
      - event_volume_new: [C,H,W]
      - flow: [2,H,W]
      - valid: [H,W] (全1)
    其中 flow 对应当前窗口的光流；下一窗口仅用于提供 event_volume_new。
    """

    def __init__(self, root_dir, num_bins=5, sequence_length=2, flow_scale=1.0, train=True, split_file=None):
        self.root = Path(root_dir)
        self.num_bins = num_bins
        self.sequence_length = max(sequence_length, 2)  # 至少两帧组成一个样本
        self.flow_scale = flow_scale
        self.train = train

        # 收集序列
        if split_file is not None:
            # 从 split 文件读取序列名称
            split_path = Path(split_file)
            if not split_path.exists():
                raise FileNotFoundError(f"Split file not found: {split_file}")
            with open(split_path, 'r') as f:
                seq_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            # 只加载 split 文件中指定的序列
            self.sequences = sorted([self.root / name for name in seq_names if (self.root / name).is_dir()])
            print(f"[DSEC] Loaded {len(self.sequences)} sequences from split file: {split_file}")
        else:
            # 加载所有序列（向后兼容）
            self.sequences = sorted([d for d in self.root.iterdir() if d.is_dir()])

        self.frames, self.windows = self._collect_samples()
        total_frames = sum(len(f) for f in self.frames)
        print(
            f"[DSEC] loaded {len(self.windows)} pairs "
            f"from {total_frames} frames in {len(self.frames)} sequences. "
            f"bins={self.num_bins}, seq_len={self.sequence_length}"
        )

    def _collect_samples(self):
        frames_all = []
        for seq_dir in self.sequences:
            event_path = seq_dir / "events" / "left" / "events.h5"
            flow_root = seq_dir / "flow"
            forward_dir = flow_root / "forward"
            ts_file = flow_root / "forward_timestamps.txt"
            flow_files = sorted(forward_dir.glob("*.png")) if forward_dir.exists() else []
            timestamps = self._load_timestamps(ts_file)

            if not event_path.exists() or len(flow_files) == 0 or timestamps is None:
                print(f"[DSEC] skip {seq_dir.name}: missing events/flow/timestamps")
                frames_all.append([])
                continue

            if len(timestamps) < len(flow_files):
                flow_files = flow_files[: len(timestamps)]

            frames = []
            for i, fp in enumerate(flow_files):
                start_us, end_us = timestamps[i]
                frames.append(
                    {
                        "event_path": event_path,
                        "flow_path": fp,
                        "start_us": start_us,
                        "end_us": end_us,
                    }
                )
            frames_all.append(frames)

        # 形成窗口 (current, next)，保证下一个存在
        windows = []
        for seq_id, frames in enumerate(frames_all):
            if len(frames) < 2:
                continue
            for i in range(0, len(frames) - 1):
                windows.append((seq_id, i))
        return frames_all, windows

    def _load_timestamps(self, ts_file: Path):
        if not ts_file.exists():
            return None
        ts = []
        with open(ts_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) != 2:
                    continue
                ts.append((int(parts[0]), int(parts[1])))
        return ts if ts else None

    def __len__(self):
        return len(self.windows)

    def _voxel_between(self, h5_path, start_us, end_us):
        with h5py.File(h5_path, "r") as h5_file:
            events = h5_file["events"]
            xs = events["x"]
            ys = events["y"]
            ts = events["t"]
            ps = events["p"]
            # 元信息：height/width/t_offset 在不同数据可能以 attrs 形式保存
            def _get_attr(src, key, default):
                if isinstance(src, dict) and key in src:
                    return src[key]
                if hasattr(src, "attrs") and key in src.attrs:
                    return src.attrs[key]
                return default

            h = int(_get_attr(h5_file, "height", _get_attr(events, "height", 480)))
            w = int(_get_attr(h5_file, "width", _get_attr(events, "width", 640)))
            t_offset = int(_get_attr(h5_file, "t_offset", _get_attr(events, "t_offset", 0)))

            start_rel = max(int(start_us - t_offset), 0)
            end_rel = max(int(end_us - t_offset), start_rel + 1)
            total = ts.shape[0]

            # 二分定位
            def bsearch(target, side):
                lo, hi = 0, total
                while lo < hi:
                    mid = (lo + hi) // 2
                    mv = int(ts[mid])
                    if side == "left":
                        if mv < target:
                            lo = mid + 1
                        else:
                            hi = mid
                    else:
                        if mv <= target:
                            lo = mid + 1
                        else:
                            hi = mid
                return lo

            s_idx = bsearch(start_rel, "left")
            e_idx = bsearch(end_rel, "right")
            if e_idx <= s_idx:
                return torch.zeros(self.num_bins, h, w)

            xs = xs[s_idx:e_idx].astype(np.int32)
            ys = ys[s_idx:e_idx].astype(np.int32)
            ts_sel = ts[s_idx:e_idx].astype(np.int64)
            ps = ps[s_idx:e_idx].astype(np.int8)
            mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            if not np.any(mask):
                return torch.zeros(self.num_bins, h, w)

            xs = xs[mask]
            ys = ys[mask]
            ts_sel = ts_sel[mask]
            ps = ps[mask]

            duration = max(end_rel - start_rel, 1)
            bin_ids = ((ts_sel - start_rel) * self.num_bins / duration).astype(np.int32)
            bin_ids = np.clip(bin_ids, 0, self.num_bins - 1)
            pol = np.where(ps > 0, 1.0, -1.0)

            voxel = np.zeros((self.num_bins, h, w), dtype=np.float32)
            np.add.at(voxel, (bin_ids, ys, xs), pol)
            # 简单截断异常值
            limit = np.quantile(np.abs(voxel), 0.999) if voxel.size > 0 else 0
            if limit > 0:
                voxel = np.clip(voxel, -limit, limit)
            return torch.from_numpy(voxel)

    def __getitem__(self, idx):
        seq_id, start_idx = self.windows[idx]
        frames = self.frames[seq_id]
        curr = frames[start_idx]
        nxt = frames[start_idx + 1]

        ev_old = self._voxel_between(curr["event_path"], curr["start_us"], curr["end_us"])
        ev_new = self._voxel_between(nxt["event_path"], nxt["start_us"], nxt["end_us"])
        flow = load_flow_png(curr["flow_path"], flow_scale=self.flow_scale)

        h, w = flow.shape[1], flow.shape[2]
        valid = torch.ones(h, w, dtype=torch.float32)

        sample = {
            "names": f"{self.sequences[seq_id].name}_{start_idx}",
            "event_volume_old": ev_old,
            "event_volume_new": ev_new,
            "flow": flow,
            "valid": valid,
        }
        return sample

