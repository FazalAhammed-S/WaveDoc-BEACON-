"""
WaveDoc — model architectures.
Must match the definitions used at training time exactly.
"""
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Core layer: ENAQT (Environment-Assisted Quantum Transport)
# ─────────────────────────────────────────────────────────────
class ENAQTLayer(nn.Module):
    """Complex-valued layer with a learnable decoherence parameter."""

    def __init__(self, in_f, out_f):
        super().__init__()
        self.W_real = nn.Parameter(torch.randn(in_f, out_f) * 0.01)
        self.W_imag = nn.Parameter(torch.randn(in_f, out_f) * 0.01)
        self.gamma = nn.Parameter(torch.ones(out_f) * 0.5)
        self.bn_r = nn.BatchNorm1d(out_f)
        self.bn_i = nn.BatchNorm1d(out_f)

    def forward(self, xr, xi):
        out_r = xr @ self.W_real - xi @ self.W_imag
        out_i = xr @ self.W_imag + xi @ self.W_real
        noise = self.gamma.abs() * torch.randn_like(out_r)
        out_r = self.bn_r(out_r + noise)
        out_i = self.bn_i(out_i)
        return out_r, out_i, torch.sqrt(out_r ** 2 + out_i ** 2 + 1e-8)

    def get_gamma(self):
        return self.gamma.abs().mean().item()


class ENAQTConvLayer(nn.Module):
    """ENAQT applied to 2D feature maps, preserving spatial extent."""

    def __init__(self, ch):
        super().__init__()
        self.W_real = nn.Conv2d(ch, ch, 3, padding=1)
        self.W_imag = nn.Conv2d(ch, ch, 3, padding=1)
        self.gamma = nn.Parameter(torch.ones(ch, 1, 1) * 0.5)
        self.bn_r = nn.BatchNorm2d(ch)
        self.bn_i = nn.BatchNorm2d(ch)

    def forward(self, xr, xi):
        r = self.W_real(xr) - self.W_imag(xi)
        im = self.W_imag(xr) + self.W_real(xi)
        noise = self.gamma.abs() * torch.randn_like(r)
        r = self.bn_r(r + noise)
        im = self.bn_i(im)
        return r, im, torch.sqrt(r ** 2 + im ** 2 + 1e-8)

    def get_gamma(self):
        return self.gamma.abs().mean().item()


def _gamma_profile(layers):
    return {f"L{i + 1}": l.get_gamma() for i, l in enumerate(layers)}


# ─────────────────────────────────────────────────────────────
# MIT-BIH: single-lead ECG beat classification
# ─────────────────────────────────────────────────────────────
class QuantumBioNet(nn.Module):
    def __init__(self, inp=360, hidden=256, layers=3, classes=5):
        super().__init__()
        sizes = [inp] + [hidden] * layers
        self.layers = nn.ModuleList(
            [ENAQTLayer(sizes[i], sizes[i + 1]) for i in range(layers)])
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, classes)
        self.act = nn.ReLU()

    def forward(self, xr, xi):
        for l in self.layers:
            xr, xi, amp = l(xr, xi)
            xr = self.act(self.drop(xr))
            xi = self.act(self.drop(xi))
        return self.fc(amp)

    def get_phase(self):
        last = self.layers[-1]
        return torch.atan2(last.W_imag, last.W_real)

    def get_gamma_profile(self):
        return _gamma_profile(self.layers)


# ─────────────────────────────────────────────────────────────
# PTB-XL: 12-lead ECG superclass diagnosis
# ─────────────────────────────────────────────────────────────
class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, 2, 2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 3, 2, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 512, 3, 2, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        return self.conv(x.view(x.shape[0], 12, 1000)).squeeze(-1)


class BEACONForPTBXL(nn.Module):
    def __init__(self, classes=5, layers=6):
        super().__init__()
        self.enc_r = ConvEncoder()
        self.enc_i = ConvEncoder()
        sizes = [512] + [512] * layers
        self.layers = nn.ModuleList(
            [ENAQTLayer(sizes[i], sizes[i + 1]) for i in range(layers)])
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(512, classes)
        self.act = nn.ReLU()

    def forward(self, xr, xi):
        xr = self.enc_r(xr)
        xi = self.enc_i(xi)
        for l in self.layers:
            xr, xi, amp = l(xr, xi)
            xr = self.act(self.drop(xr))
            xi = self.act(self.drop(xi))
        return self.fc(amp)

    def get_phase(self):
        last = self.layers[-1]
        return torch.atan2(last.W_imag, last.W_real)

    def get_gamma_profile(self):
        return _gamma_profile(self.layers)


# ─────────────────────────────────────────────────────────────
# EEG: sleep staging and seizure detection
# ─────────────────────────────────────────────────────────────
class EEGConvEncoder(nn.Module):
    """Two-branch encoder: short kernels for transients, long for rhythms."""

    def __init__(self, ch):
        super().__init__()
        self.branch_fast = nn.Sequential(
            nn.Conv1d(ch, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, 2, 2), nn.BatchNorm1d(128), nn.ReLU())
        self.branch_slow = nn.Sequential(
            nn.Conv1d(ch, 64, 25, 2, 12), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 15, 2, 7), nn.BatchNorm1d(128), nn.ReLU())
        self.fuse = nn.Sequential(
            nn.Conv1d(256, 256, 3, 2, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 512, 3, 2, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        f = self.branch_fast(x)
        s = self.branch_slow(x)
        m = min(f.shape[-1], s.shape[-1])
        return self.fuse(torch.cat([f[..., :m], s[..., :m]], 1)).squeeze(-1)


class BEACONForEEG(nn.Module):
    def __init__(self, ch, classes, layers=6):
        super().__init__()
        self.enc_r = EEGConvEncoder(ch)
        self.enc_i = EEGConvEncoder(ch)
        sizes = [512] + [512] * layers
        self.layers = nn.ModuleList(
            [ENAQTLayer(sizes[i], sizes[i + 1]) for i in range(layers)])
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, classes))
        self.act = nn.ReLU()

    def forward(self, xr, xi):
        xr = self.enc_r(xr)
        xi = self.enc_i(xi)
        for l in self.layers:
            xr, xi, amp = l(xr, xi)
            xr = self.act(self.drop(xr))
            xi = self.act(self.drop(xi))
        return self.fc(amp)

    def get_phase(self):
        last = self.layers[-1]
        return torch.atan2(last.W_imag, last.W_real)

    def get_gamma_profile(self):
        return _gamma_profile(self.layers)


# ─────────────────────────────────────────────────────────────
# CAMUS: cardiac ultrasound segmentation
# ─────────────────────────────────────────────────────────────
def _conv_block(cin, cout, dropout=0.0):
    layers = [
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.1, inplace=True)]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class BEACONSegUSG(nn.Module):
    """U-Net with an ENAQT bottleneck. Outputs a full-resolution mask."""

    def __init__(self, num_classes=4, base=32):
        super().__init__()
        b = base
        self.e1r = _conv_block(1, b);          self.e1i = _conv_block(1, b)
        self.e2r = _conv_block(b, b * 2);      self.e2i = _conv_block(b, b * 2)
        self.e3r = _conv_block(b * 2, b * 4, 0.1)
        self.e3i = _conv_block(b * 2, b * 4, 0.1)
        self.e4r = _conv_block(b * 4, b * 8, 0.2)
        self.e4i = _conv_block(b * 4, b * 8, 0.2)
        self.pool = nn.MaxPool2d(2)

        self.enaqt1 = ENAQTConvLayer(b * 8)
        self.enaqt2 = ENAQTConvLayer(b * 8)
        self.enaqt3 = ENAQTConvLayer(b * 8)
        self.enaqt4 = ENAQTConvLayer(b * 8)

        self.up4 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2)
        self.dec4 = _conv_block(b * 8, b * 4, 0.1)
        self.up3 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2)
        self.dec3 = _conv_block(b * 4, b * 2)
        self.up2 = nn.ConvTranspose2d(b * 2, b, 2, 2)
        self.dec2 = _conv_block(b * 2, b)
        self.final = nn.Conv2d(b, num_classes, 1)

    def _encode(self, xr, xi):
        e1r = self.e1r(xr); p1r = self.pool(e1r)
        e2r = self.e2r(p1r); p2r = self.pool(e2r)
        e3r = self.e3r(p2r); p3r = self.pool(e3r)
        e4r = self.e4r(p3r)
        e1i = self.e1i(xi); p1i = self.pool(e1i)
        e2i = self.e2i(p1i); p2i = self.pool(e2i)
        e3i = self.e3i(p2i); p3i = self.pool(e3i)
        e4i = self.e4i(p3i)
        return (e1r, e2r, e3r, e4r), (e4i,)

    def forward(self, xr, xi):
        (e1r, e2r, e3r, e4r), (e4i,) = self._encode(xr, xi)
        r, im, amp = self.enaqt1(e4r, e4i)
        r, im, amp = self.enaqt2(r, im)
        r, im, amp = self.enaqt3(r, im)
        r, im, amp = self.enaqt4(r, im)
        d4 = self.up4(amp); d4 = torch.cat([d4, e3r], 1); d4 = self.dec4(d4)
        d3 = self.up3(d4);  d3 = torch.cat([d3, e2r], 1); d3 = self.dec3(d3)
        d2 = self.up2(d3);  d2 = torch.cat([d2, e1r], 1); d2 = self.dec2(d2)
        return self.final(d2)

    @torch.no_grad()
    def get_phase_spatial(self, xr, xi):
        """Phase map registered to the input frame."""
        (_, _, _, e4r), (e4i,) = self._encode(xr, xi)
        r, im, _ = self.enaqt1(e4r, e4i)
        r, im, _ = self.enaqt2(r, im)
        r, im, _ = self.enaqt3(r, im)
        r, im, _ = self.enaqt4(r, im)
        return torch.atan2(im, r).mean(1)[0]

    def get_phase(self):
        w = self.enaqt4
        return torch.atan2(w.W_imag.weight[:, :, 1, 1],
                           w.W_real.weight[:, :, 1, 1])

    def get_gamma_profile(self):
        return {"L1": self.enaqt1.get_gamma(), "L2": self.enaqt2.get_gamma(),
                "L3": self.enaqt3.get_gamma(), "L4": self.enaqt4.get_gamma()}
