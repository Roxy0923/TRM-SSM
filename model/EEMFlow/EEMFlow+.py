import os,sys
current_path = os.path.dirname(os.path.abspath(__file__))
proc_path = current_path.rsplit("/",1)[0]
sys.path.append(current_path)
sys.path.append(proc_path)
# Add RVT models to path for temporal SSM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler
from utils_luo.tools import tools, tensor_tools
from utils.image_utils import ImagePadder, InputPadder
from model.EEMFlow.cdc_utils import cdc_model, cdc_ssm_model, conv, upsample2d_flow_as
import pdb

# Import Temporal SSM (optional, with fallback)
try:
    from RVT.models.temporal_ssm import TemporalSSMPatchwise, SpatioTemporalSSM
    TEMPORAL_SSM_AVAILABLE = True
except ImportError:
    TEMPORAL_SSM_AVAILABLE = False
    print("[Warning] TemporalSSM not available. Install RVT models to enable.")


class Correlation(nn.Module):
    def __init__(self, max_displacement):
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2*max_displacement+1
        self.corr = SpatialCorrelationSampler(1, self.kernel_size, 1, 0, 1)
        
    def forward(self, x, y):
        b, c, h, w = x.shape
        return self.corr(x, y).view(b, -1, h, w) / c


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.LeakyReLU(0.1, inplace=True)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class Decoder(nn.Module):
    def __init__(self, in_channels, groups):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = convrelu(in_channels, 96, 3, 1)
        self.conv2 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv3 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv4 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv5 = convrelu(96, 64, 3, 1)
        self.conv6 = convrelu(64, 32, 3, 1)
        self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)


    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x


    def forward(self, x):
        if self.groups == 1:
            out = self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))))
        else:
            out = self.conv1(x)
            out = self.channel_shuffle(self.conv2(out), self.groups)
            out = self.channel_shuffle(self.conv3(out), self.groups)
            out = self.channel_shuffle(self.conv4(out), self.groups)
            out = self.conv7(self.conv6(self.conv5(out)))
        return out


class EEMFlow_cdc(nn.Module):
    def __init__(self, config, groups=3, n_first_channels=15, args=None):
        super(EEMFlow_cdc, self).__init__()
        self.args = args
        self.groups = groups
        self.n_first_channels = n_first_channels

        # ============ Temporal SSM Configuration (NEW) ============
        # Check if temporal SSM is enabled
        self.use_temporal_ssm = False
        self.temporal_step_scale = 1.0  # Default step_scale for temporal SSM
        self.temporal_debug_iters = getattr(args, 'temporal_debug_iters', 0) if args is not None else 0
        self._temporal_debug_count = 0

        if args is not None:
            self.use_temporal_ssm = getattr(args, 'use_temporal_ssm', False)
            self.temporal_step_scale = getattr(args, 'step_scale', 1.0)

        # Initialize Temporal SSM if enabled
        if self.use_temporal_ssm:
            if not TEMPORAL_SSM_AVAILABLE:
                raise RuntimeError(
                    "Temporal SSM requested but not available. "
                    "Please check RVT/models/temporal_ssm.py exists."
                )

            temporal_state_dim = getattr(args, 'temporal_state_dim', 32) if args else 32
            temporal_out_channels = 16  # Match pconv1_1 input expectation
            # Use patchify strategy for memory efficiency while preserving spatial info
            patch_size = 8  # 8x8 patches: reduces 480x640 to 60x80 = 4800 tokens
            d_model = 64   # Internal feature dimension

            print(f"[EEMFlow] Enabling Temporal SSM (Patchwise):")
            print(f"  - step_scale={self.temporal_step_scale} (dt1=1.0, dt4=4.0)")
            print(f"  - patch_size={patch_size}x{patch_size}")
            print(f"  - d_model={d_model}, state_dim={temporal_state_dim}")
            print(f"  - in_channels={n_first_channels}, out_channels={temporal_out_channels}")

            self.temporal_ssm = TemporalSSMPatchwise(
                in_channels=n_first_channels,
                out_channels=temporal_out_channels,
                patch_size=patch_size,
                d_model=d_model,
                state_dim=temporal_state_dim,
                dropout=0.0,
            )
            # Adjust first conv to accept temporal SSM output
            encoder_in_channels = temporal_out_channels
        else:
            self.temporal_ssm = None
            encoder_in_channels = n_first_channels

        # ============ Encoder ============
        self.pconv1_1 = convrelu(encoder_in_channels, 16, 3, 2)
        self.pconv1_2 = convrelu(16, 16, 3, 1)
        self.pconv2_1 = convrelu(16, 32, 3, 2)
        self.pconv2_2 = convrelu(32, 32, 3, 1)
        self.pconv2_3 = convrelu(32, 32, 3, 1)
        self.pconv3_1 = convrelu(32, 64, 3, 2)
        self.pconv3_2 = convrelu(64, 64, 3, 1)
        self.pconv3_3 = convrelu(64, 64, 3, 1)

        self.corr = Correlation(4)
        self.index = torch.tensor([0, 2, 4, 6, 8, 
                10, 12, 14, 16, 
                18, 20, 21, 22, 23, 24, 26, 
                28, 29, 30, 31, 32, 33, 34, 
                36, 38, 39, 40, 41, 42, 44, 
                46, 47, 48, 49, 50, 51, 52, 
                54, 56, 57, 58, 59, 60, 62, 
                64, 66, 68, 70, 
                72, 74, 76, 78, 80])

        self.rconv2 = convrelu(32, 32, 3, 1)
        self.rconv3 = convrelu(64, 32, 3, 1)
        self.rconv4 = convrelu(64, 32, 3, 1)
        self.rconv5 = convrelu(64, 32, 3, 1)
        self.rconv6 = convrelu(64, 32, 3, 1)

        self.up3 = deconv(2, 2)
        self.up4 = deconv(2, 2)
        self.up5 = deconv(2, 2)
        self.up6 = deconv(2, 2)

        self.decoder2 = Decoder(87, groups)
        self.decoder3 = Decoder(87, groups)
        self.decoder4 = Decoder(87, groups)
        self.decoder5 = Decoder(87, groups)
        self.decoder6 = Decoder(87, groups)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # self.image_padder = ImagePadder(min_size=64)

        # **NEW: Support both cdc_model and cdc_ssm_model based on args**
        if args is not None and hasattr(args, 'use_ssm') and args.use_ssm:
            # Use SSM-based CDC with frequency adaptation
            # IMPORTANT: Spatial SSM always uses step_scale=1.0!
            # The step_scale for frequency adaptation only applies to Temporal SSM
            spatial_step_scale = 1.0  # Fixed - spatial resolution is same for dt1/dt4
            ssm_state_dim = getattr(args, 'ssm_state_dim', 64)
            blend_weight = getattr(args, 'blend_weight', 0.3)
            ssm_dropout = getattr(args, 'ssm_dropout', 0.1)
            h2_reg_weight = getattr(args, 'h2_reg_weight', 0.0)

            print(f"[EEMFlow] Using cdc_ssm_model with spatial_step_scale={spatial_step_scale} (FIXED), "
                  f"state_dim={ssm_state_dim}, blend_weight={blend_weight}, h2_reg_weight={h2_reg_weight}")
            import sys
            sys.stdout.flush()  # Ensure output is immediately visible
            self.cdc_model = cdc_ssm_model(
                step_scale=spatial_step_scale,  # Always 1.0 for spatial SSM!
                ssm_state_dim=ssm_state_dim,
                blend_weight=blend_weight,
                dropout=ssm_dropout,
                h2_reg_weight=h2_reg_weight,
            )
        else:
            # Use baseline CDC model
            print("[EEMFlow] Using baseline cdc_model")
            self.cdc_model = cdc_model()

        self.conv_1x1 = nn.ModuleList([
                                conv(15, 32, kernel_size=1, stride=1, dilation=1),
                                conv(16, 32, kernel_size=1, stride=1, dilation=1),
                                conv(32, 32, kernel_size=1, stride=1, dilation=1),
                                conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                conv(64, 32, kernel_size=1, stride=1, dilation=1)])

    def change_imagesize(self, img_size):
        self.image_size = img_size
        self.image_padder = InputPadder(img_size, mode='chairs', eval_pad_rate=64)

    def warp(self, x, flo):
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat([xx, yy], 1).to(x)
        vgrid = grid + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H-1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)        
        output = F.grid_sample(x, vgrid, mode='bilinear', align_corners=True)
        return output
    
    def self_guided_upsample(self, flow_up_bilinear, feature_1, feature_2, output_level_flow=None, return_inter=True):
        flow_up_bilinear_, out_flow, inter_flow, inter_mask = self.cdc_model(flow_up_bilinear, feature_1, feature_2, output_level_flow=output_level_flow)
        if(return_inter):
            return out_flow
        else:
            return out_flow, inter_flow, inter_mask

    def forward(self, events1, events2, step_scale=None):
        """
        Forward pass with optional temporal SSM.

        Args:
            events1: [B, T, H, W] - Event tensor for frame 1 (T=num_bins, e.g., 15)
            events2: [B, T, H, W] - Event tensor for frame 2
            step_scale: float (optional) - Override temporal step_scale for inference
                Based on S5 paper: step_scale = reference_freq / data_freq
                - 1.0 for 60Hz (dt1, reference)
                - 4.0 for 15Hz (dt4)
                - If None, use self.temporal_step_scale

        Returns:
            (events1, events2), flow_predictions
        """
        # Use provided step_scale or default
        current_step_scale = step_scale if step_scale is not None else self.temporal_step_scale

        # ============ Temporal SSM (if enabled) ============
        if self.use_temporal_ssm and self.temporal_ssm is not None:
            # Apply temporal SSM to model dependencies across bins
            # step_scale here has physical meaning: temporal discretization
            events1_encoded = self.temporal_ssm(events1, step_scale=current_step_scale)
            events2_encoded = self.temporal_ssm(events2, step_scale=current_step_scale)
            # events_encoded: [B, out_channels, H, W]
            if self.temporal_debug_iters and self._temporal_debug_count < self.temporal_debug_iters:
                with torch.no_grad():
                    def _stat(x):
                        return x.mean().item(), x.std().item(), x.abs().max().item()
                    m1, s1, mx1 = _stat(events1_encoded)
                    m2, s2, mx2 = _stat(events2_encoded)
                    print(f"[TemporalSSM][debug {self._temporal_debug_count}] "
                          f"step_scale={float(current_step_scale):.3f} "
                          f"e1 mean={m1:.4f} std={s1:.4f} max={mx1:.4f} "
                          f"e2 mean={m2:.4f} std={s2:.4f} max={mx2:.4f} "
                          f"shape={tuple(events1_encoded.shape)}", flush=True)
                self._temporal_debug_count += 1
        else:
            # No temporal SSM, use raw events
            events1_encoded = events1
            events2_encoded = events2

        # ============ Padding ============
        if not hasattr(self, "image_padder") or self.image_padder is None:
            # Lazily init padder to match current input size
            self.change_imagesize((events1_encoded.shape[-2], events1_encoded.shape[-1]))
        image1, image2 = self.image_padder.pad(events1_encoded, events2_encoded)

        f11 = self.pconv1_2(self.pconv1_1(image1))
        f21 = self.pconv1_2(self.pconv1_1(image2))
        f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
        f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
        f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
        f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
        f14 = F.avg_pool2d(f13, kernel_size=(2, 2), stride=(2, 2))
        f24 = F.avg_pool2d(f23, kernel_size=(2, 2), stride=(2, 2))
        f15 = F.avg_pool2d(f14, kernel_size=(2, 2), stride=(2, 2))
        f25 = F.avg_pool2d(f24, kernel_size=(2, 2), stride=(2, 2))
        f16 = F.avg_pool2d(f15, kernel_size=(2, 2), stride=(2, 2))
        f26 = F.avg_pool2d(f25, kernel_size=(2, 2), stride=(2, 2))

        flow7_up = torch.zeros(f16.size(0), 2, f16.size(2), f16.size(3)).to(f15)
        cv6 = torch.index_select(self.corr(f16, f26), dim=1, index=self.index.to(f16).long())
        r16 = self.rconv6(f16)
        cat6 = torch.cat([cv6, r16, flow7_up], 1)
        flow6 = self.decoder6(cat6)

        l=5
        f15_1x1 = self.conv_1x1[l](f15)
        f25_1x1 = self.conv_1x1[l](f25)
        flow6_up = self.self_guided_upsample(flow6, f15_1x1, f25_1x1)
        # flow6_up = self.up6(flow6)

        f25_w = self.warp(f25, flow6_up)
        cv5 = torch.index_select(self.corr(f15, f25_w), dim=1, index=self.index.to(f15).long())
        r15 = self.rconv5(f15)
        cat5 = torch.cat([cv5, r15, flow6_up], 1)
        flow5 = self.decoder5(cat5) + flow6_up

        l=4
        f14_1x1 = self.conv_1x1[l](f14)
        f24_1x1 = self.conv_1x1[l](f24)
        flow5_up = self.self_guided_upsample(flow5, f14_1x1, f24_1x1)
        # flow5_up = self.up5(flow5)
        
        f24_w = self.warp(f24, flow5_up)
        cv4 = torch.index_select(self.corr(f14, f24_w), dim=1, index=self.index.to(f14).long())
        r14 = self.rconv4(f14)
        cat4 = torch.cat([cv4, r14, flow5_up], 1)
        flow4 = self.decoder4(cat4) + flow5_up

        l=3
        f13_1x1 = self.conv_1x1[l](f13)
        f23_1x1 = self.conv_1x1[l](f23)
        flow4_up = self.self_guided_upsample(flow4, f13_1x1, f23_1x1)
        # flow4_up = self.up4(flow4)

        f23_w = self.warp(f23, flow4_up)
        cv3 = torch.index_select(self.corr(f13, f23_w), dim=1, index=self.index.to(f13).long())
        r13 = self.rconv3(f13)
        cat3 = torch.cat([cv3, r13, flow4_up], 1)
        flow3 = self.decoder3(cat3) + flow4_up

        l=2
        f12_1x1 = self.conv_1x1[l](f12)
        f22_1x1 = self.conv_1x1[l](f22)
        flow3_up = self.self_guided_upsample(flow3, f12_1x1, f22_1x1)
        # flow3_up = self.up3(flow3)

        f22_w = self.warp(f22, flow3_up)
        cv2 = torch.index_select(self.corr(f12, f22_w), dim=1, index=self.index.to(f12).long())
        r12 = self.rconv2(f12)
        cat2 = torch.cat([cv2, r12, flow3_up], 1)
        flow2 = self.decoder2(cat2) + flow3_up

        flow_predictions = [upsample2d_flow_as(flow6, events1, mode="bilinear", if_rate=True), upsample2d_flow_as(flow5, events1, mode="bilinear", if_rate=True), upsample2d_flow_as(flow4, events1, mode="bilinear", if_rate=True),
            upsample2d_flow_as(flow3, events1, mode="bilinear", if_rate=True), upsample2d_flow_as(flow2, events1, mode="bilinear", if_rate=True)]

        return (events1, events2), flow_predictions
            
    @classmethod
    def demo(cls):
        h, w = 480, 640
        im = torch.ones((2, 15, h, w))
        net = EEMFlow_cdc(groups=3, n_first_channels=15)
        im = im.cuda()
        net = net.cuda()

        with torch.no_grad():
            out = net(im,im)

        tensor_tools.check_tensor(out[0], 'flow')
        tensor_tools.check_tensor(out[1], 'flow2')
        tensor_tools.check_tensor(out[2], 'flow3')
        tensor_tools.check_tensor(out[3], 'flow4')
        tensor_tools.check_tensor(out[4], 'flow5')
        tensor_tools.check_tensor(out[5], 'flow6')


def time_eval(model, batch_size=1, iters=32):  # 12
    import time

    h,w = 480, 640
    # h,w = 720, 1280

    model = model.cuda()
    model = model.eval()
    images = torch.randn(batch_size, 15, h,w).cuda()
    first = 0
    time_train = []
    for ii in range(100):
        start_time = time.time()
        # model.change_imagesize((h,w))
        with torch.no_grad():
            outputs = model(images, images)
        torch.cuda.synchronize() 

        if first != 0:    #first run always takes some time for setup
            fwt = time.time() - start_time
            time_train.append(fwt)
            print ("Forward time per img (b=%d): %.4f (Mean: %.4f), FPS: %.1f" % (batch_size, fwt / batch_size, sum(time_train) / len(time_train) / batch_size, 1/(fwt/batch_size)))
        
        time.sleep(1) 
        first += 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='EEMFlow')
    #training setting
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--checkpoint_dir', type=str, default='train_TMA')
    parser.add_argument('--lr', type=float, default=2e-4)

    #datasets setting
    parser.add_argument('--root', type=str, default='dsec/')
    parser.add_argument('--crop_size', type=list, default=[288, 384])

    #dataloader setting
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=8)

    #model setting
    parser.add_argument('--grad_clip', type=float, default=1)

    # loss setting
    parser.add_argument('--weight', type=float, default=0.8)

    #wandb setting
    parser.add_argument('--wandb', action='store_true', default=False)

    parser.add_argument('--upsample_mask', type=bool, default=False)

    parser.add_argument('--cdc_model', type=str, default='cdc_model')

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = EEMFlow_cdc(config="", groups=1, n_first_channels=15, args=args)
    time_eval(model, batch_size=1)
