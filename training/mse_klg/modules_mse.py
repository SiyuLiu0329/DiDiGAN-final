"""Generator architecture from the paper
"Alias-Free Generative Adversarial Networks"."""

import numpy as np
import scipy.signal
import scipy.optimize
import torch
import torch.nn.functional as F
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import filtered_lrelu
from torch_utils.ops import bias_act
from torch_utils.ops import upfirdn2d
from torch_utils.ops import conv2d_resample


@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        activation="linear",
        up=1,
        down=1,
        resample_filter=[1, 3, 3, 1],
        conv_clamp=None,
        channels_last=False,
        trainable=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(
            memory_format=memory_format
        )
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer("weight", weight)
            if bias is not None:
                self.register_buffer("bias", bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = self.up == 1
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=w.to(x.dtype),
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
            flip_weight=flip_weight,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return " ".join(
            [
                f"in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},",
                f"up={self.up}, down={self.down}",
            ]
        )


@misc.profiled_function
def modulated_conv2d(
    x,
    w,
    s,
    demodulate=True,
    padding=0,
    input_gain=None,
):
    with misc.suppress_tracer_warnings():
        batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    misc.assert_shape(w, [out_channels, in_channels, kh, kw])
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    misc.assert_shape(s, [batch_size, in_channels])

    if demodulate:
        w = w * w.square().mean([1, 2, 3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    w = w.unsqueeze(0)
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4)

    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4)

    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels)
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4)

    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_gradfix.conv2d(
        input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size
    )
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation="linear",
        bias=True,
        lr_multiplier=1,
        weight_init=1,
        bias_init=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) * (weight_init / lr_multiplier)
        )
        bias_init = np.broadcast_to(
            np.asarray(bias_init, dtype=np.float32), [out_features]
        )
        self.bias = (
            torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier))
            if bias
            else None
        )
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        if self.activation == "linear" and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f"in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}"


@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(
        self,
        z_dim,
        c_dim,
        w_dim,
        num_ws,
        num_layers=2,
        lr_multiplier=0.01,
        w_avg_beta=0.998,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = 512
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        self.c_emb = torch.nn.Embedding(64, self.c_dim)

        self.embed = (
            FullyConnectedLayer(self.c_dim, self.w_dim) if self.c_dim > 0 else None
        )
        features = [self.z_dim + (self.w_dim if self.c_dim > 0 else 0)] + [
            self.w_dim
        ] * self.num_layers
        for idx, in_features, out_features in zip(
            range(num_layers), features[:-1], features[1:]
        ):
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation="lrelu",
                lr_multiplier=lr_multiplier,
            )
            setattr(self, f"fc{idx}", layer)
        self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(
        self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False
    ):
        c = self.c_emb(c.long())

        misc.assert_shape(z, [None, self.z_dim])
        if truncation_cutoff is None:
            truncation_cutoff = self.num_ws

        x = z.to(torch.float32)
        x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()
        if self.c_dim > 0:
            misc.assert_shape(c, [None, self.c_dim])
            y = self.embed(c.to(torch.float32))
            y = y * (y.square().mean(1, keepdim=True) + 1e-8).rsqrt()
            x = torch.cat([x, y], dim=1) if x is not None else y

        for idx in range(self.num_layers):
            x = getattr(self, f"fc{idx}")(x)

        if update_emas:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
        if truncation_psi != 1:
            x[:, :truncation_cutoff] = self.w_avg.lerp(
                x[:, :truncation_cutoff], truncation_psi
            )
        return x

    def extra_repr(self):
        return f"z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}"


@persistence.persistent_class
class SynthesisInput(torch.nn.Module):
    def __init__(
        self,
        w_dim,
        channels,
        size,
        sampling_rate,
        bandwidth,
    ):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5

        self.weight = torch.nn.Parameter(torch.randn([self.channels, self.channels]))
        self.affine = FullyConnectedLayer(
            w_dim, 4, weight_init=0, bias_init=[1, 0, 0, 0]
        )
        self.register_buffer("transform", torch.eye(3, 3))
        self.register_buffer("freqs", freqs)
        self.register_buffer("phases", phases)

    def forward(self, w):
        transforms = self.transform.unsqueeze(0)
        freqs = self.freqs.unsqueeze(0)
        phases = self.phases.unsqueeze(0)

        t = self.affine(w)
        t = t / t[:, :2].norm(dim=1, keepdim=True)
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1])
        m_r[:, 0, 0] = t[:, 0]
        m_r[:, 0, 1] = -t[:, 1]
        m_r[:, 1, 0] = t[:, 1]
        m_r[:, 1, 1] = t[:, 0]
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1])
        m_t[:, 0, 2] = -t[:, 2]
        m_t[:, 1, 2] = -t[:, 3]
        transforms = m_r @ m_t @ transforms

        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        amplitudes = (
            1
            - (freqs.norm(dim=2) - self.bandwidth)
            / (self.sampling_rate / 2 - self.bandwidth)
        ).clamp(0, 1)

        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = torch.nn.functional.affine_grid(
            theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False
        )

        x = (
            grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)
        ).squeeze(3)
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)

        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()

        x = x.permute(0, 3, 1, 2)
        misc.assert_shape(
            x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])]
        )
        return x

    def extra_repr(self):
        return "\n".join(
            [
                f"w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},",
                f"sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}",
            ]
        )


@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        w_dim,
        is_torgb,
        is_critically_sampled,
        use_fp16,
        in_channels,
        out_channels,
        in_size,
        out_size,
        in_sampling_rate,
        out_sampling_rate,
        in_cutoff,
        out_cutoff,
        in_half_width,
        out_half_width,
        conv_kernel=3,
        filter_size=6,
        lrelu_upsampling=2,
        use_radial_filters=False,
        conv_clamp=256,
        magnitude_ema_beta=0.999,
    ):
        super().__init__()
        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_critically_sampled = is_critically_sampled
        self.use_fp16 = use_fp16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (
            1 if is_torgb else lrelu_upsampling
        )
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = 1 if is_torgb else conv_kernel
        self.conv_clamp = conv_clamp
        self.magnitude_ema_beta = magnitude_ema_beta

        self.affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(
            torch.randn(
                [
                    self.out_channels,
                    self.in_channels,
                    self.conv_kernel,
                    self.conv_kernel,
                ]
            )
        )
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.register_buffer("magnitude_ema", torch.ones([]))

        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        self.up_taps = (
            filter_size * self.up_factor
            if self.up_factor > 1 and not self.is_torgb
            else 1
        )
        self.register_buffer(
            "up_filter",
            self.design_lowpass_filter(
                numtaps=self.up_taps,
                cutoff=self.in_cutoff,
                width=self.in_half_width * 2,
                fs=self.tmp_sampling_rate,
            ),
        )

        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = (
            filter_size * self.down_factor
            if self.down_factor > 1 and not self.is_torgb
            else 1
        )
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer(
            "down_filter",
            self.design_lowpass_filter(
                numtaps=self.down_taps,
                cutoff=self.out_cutoff,
                width=self.out_half_width * 2,
                fs=self.tmp_sampling_rate,
                radial=self.down_radial,
            ),
        )

        pad_total = (self.out_size - 1) * self.down_factor + 1
        pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor
        pad_total += self.up_taps + self.down_taps - 2
        pad_lo = (pad_total + self.up_factor) // 2
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

    def forward(self, x, w, noise_mode="random", force_fp32=False, update_emas=False):
        assert noise_mode in ["random", "const", "none"]
        misc.assert_shape(
            x, [None, self.in_channels, int(self.in_size[1]), int(self.in_size[0])]
        )
        misc.assert_shape(w, [x.shape[0], self.w_dim])

        if update_emas:
            with torch.autograd.profiler.record_function("update_magnitude_ema"):
                magnitude_cur = x.detach().to(torch.float32).square().mean()
                self.magnitude_ema.copy_(
                    magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta)
                )
        input_gain = self.magnitude_ema.rsqrt()

        styles = self.affine(w)
        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel**2))
            styles = styles * weight_gain

        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )
        x = modulated_conv2d(
            x=x.to(dtype),
            w=self.weight,
            s=styles,
            padding=self.conv_kernel - 1,
            demodulate=(not self.is_torgb),
            input_gain=input_gain,
        )

        gain = 1 if self.is_torgb else np.sqrt(2)
        slope = 1 if self.is_torgb else 0.2
        x = filtered_lrelu.filtered_lrelu(
            x=x,
            fu=self.up_filter,
            fd=self.down_filter,
            b=self.bias.to(x.dtype),
            up=self.up_factor,
            down=self.down_factor,
            padding=self.padding,
            gain=gain,
            slope=slope,
            clamp=self.conv_clamp,
        )

        misc.assert_shape(
            x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])]
        )
        assert x.dtype == dtype
        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        assert numtaps >= 1

        if numtaps == 1:
            return None

        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)

        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(
            scipy.signal.kaiser_atten(numtaps, width / (fs / 2))
        )
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return torch.as_tensor(f, dtype=torch.float32)

    def extra_repr(self):
        return "\n".join(
            [
                f"w_dim={self.w_dim:d}, is_torgb={self.is_torgb},",
                f"is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},",
                f"in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},",
                f"in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},",
                f"in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},",
                f"in_size={list(self.in_size)}, out_size={list(self.out_size)},",
                f"in_channels={self.in_channels:d}, out_channels={self.out_channels:d}",
            ]
        )


@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(
        self,
        w_dim,
        img_resolution,
        img_channels,
        channel_base=32768,
        channel_max=512,
        num_layers=14,
        num_critical=2,
        first_cutoff=2,
        first_stopband=2**2.1,
        last_stopband_rel=2**0.3,
        margin_size=10,
        output_scale=0.25,
        num_fp16_res=4,
        **layer_kwargs,
    ):
        super().__init__()
        self.w_dim = w_dim
        self.num_ws = num_layers + 2
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_layers = num_layers
        self.num_critical = num_critical
        self.margin_size = margin_size
        self.output_scale = output_scale
        self.num_fp16_res = num_fp16_res

        last_cutoff = self.img_resolution / 2
        last_stopband = last_cutoff * last_stopband_rel
        exponents = np.minimum(
            np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1
        )
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents

        sampling_rates = np.exp2(
            np.ceil(np.log2(np.minimum(stopbands * 2, self.img_resolution)))
        )
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs
        sizes = sampling_rates + self.margin_size * 2
        sizes[-2:] = self.img_resolution
        channels = np.rint(np.minimum((channel_base / 2) / cutoffs, channel_max))
        channels[-1] = self.img_channels

        self.input = SynthesisInput(
            w_dim=self.w_dim,
            channels=int(channels[0]),
            size=int(sizes[0]),
            sampling_rate=sampling_rates[0],
            bandwidth=cutoffs[0],
        )
        self.layer_names = []
        for idx in range(self.num_layers + 1):
            prev = max(idx - 1, 0)
            is_torgb = idx == self.num_layers
            is_critically_sampled = idx >= self.num_layers - self.num_critical
            use_fp16 = (
                sampling_rates[idx] * (2**self.num_fp16_res) > self.img_resolution
            )
            layer = SynthesisLayer(
                w_dim=self.w_dim,
                is_torgb=is_torgb,
                is_critically_sampled=is_critically_sampled,
                use_fp16=use_fp16,
                in_channels=int(channels[prev] + 1),
                out_channels=int(channels[idx]),
                in_size=int(sizes[prev]),
                out_size=int(sizes[idx]),
                in_sampling_rate=int(sampling_rates[prev]),
                out_sampling_rate=int(sampling_rates[idx]),
                in_cutoff=cutoffs[prev],
                out_cutoff=cutoffs[idx],
                in_half_width=half_widths[prev],
                out_half_width=half_widths[idx],
                **layer_kwargs,
            )
            name = f"L{idx}_{layer.out_size[0]}_{layer.out_channels}"
            setattr(self, name, layer)
            self.layer_names.append(name)

    def forward(self, ws, c, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        ws = ws.to(torch.float32).unbind(dim=1)

        x = self.input(ws[0])

        for name, w in zip(self.layer_names, ws[1:]):
            cx = F.interpolate(c, x.shape[2:], mode="nearest")
            x = torch.cat([cx, x], 1)
            x = getattr(self, name)(x, w, **layer_kwargs)

        if self.output_scale != 1:
            x = x * self.output_scale

        misc.assert_shape(
            x, [None, self.img_channels, self.img_resolution, self.img_resolution]
        )
        x = x.to(torch.float32)
        return x

    def extra_repr(self):
        return "\n".join(
            [
                f"w_dim={self.w_dim:d}, num_ws={self.num_ws:d},",
                f"img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},",
                f"num_layers={self.num_layers:d}, num_critical={self.num_critical:d},",
                f"margin_size={self.margin_size:d}, num_fp16_res={self.num_fp16_res:d}",
            ]
        )


@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(
        self,
        z_dim,
        c_dim,
        w_dim,
        img_resolution,
        img_channels,
        mapping_kwargs={},
        **synthesis_kwargs,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs,
        )
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(
            z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs
        )

    def forward(
        self,
        z,
        c,
        clz,
        truncation_psi=1,
        truncation_cutoff=None,
        update_emas=False,
        **synthesis_kwargs,
    ):
        ws = self.mapping(
            z,
            clz,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            update_emas=update_emas,
        )
        img = self.synthesis(ws, c, update_emas=update_emas, **synthesis_kwargs)
        return img


@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        tmp_channels,
        out_channels,
        resolution,
        img_channels,
        first_layer_idx,
        architecture="resnet",
        activation="lrelu",
        resample_filter=[1, 3, 3, 1],
        conv_clamp=None,
        use_fp16=False,
        fp16_channels_last=False,
        freeze_layers=0,
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0

        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = layer_idx >= freeze_layers
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == "skip":
            self.fromrgb = Conv2dLayer(
                img_channels,
                tmp_channels,
                kernel_size=1,
                activation=activation,
                trainable=next(trainable_iter),
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
            )

        self.conv0 = Conv2dLayer(
            tmp_channels,
            tmp_channels,
            kernel_size=3,
            activation=activation,
            trainable=next(trainable_iter),
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        self.conv1 = Conv2dLayer(
            tmp_channels,
            out_channels,
            kernel_size=3,
            activation=activation,
            down=2,
            trainable=next(trainable_iter),
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        if architecture == "resnet":
            self.skip = Conv2dLayer(
                tmp_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                down=2,
                trainable=next(trainable_iter),
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(self, x, img, force_fp32=False):
        if (x if x is not None else img).device.type != "cuda":
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = (
            torch.channels_last
            if self.channels_last and not force_fp32
            else torch.contiguous_format
        )

        if x is not None:
            misc.assert_shape(
                x, [None, self.in_channels, self.resolution, self.resolution]
            )
            x = x.to(dtype=dtype, memory_format=memory_format)

        if self.in_channels == 0 or self.architecture == "skip":
            misc.assert_shape(
                img, [None, self.img_channels, self.resolution, self.resolution]
            )
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = (
                upfirdn2d.downsample2d(img, self.resample_filter)
                if self.architecture == "skip"
                else None
            )

        if self.architecture == "resnet":
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

    def extra_repr(self):
        return f"resolution={self.resolution:d}, architecture={self.architecture:s}"


@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings():
            G = (
                torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N))
                if self.group_size is not None
                else N
            )
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)
        y = y - y.mean(dim=0)
        y = y.square().mean(dim=0)
        y = (y + 1e-8).sqrt()
        y = y.mean(dim=[2, 3, 4])
        y = y.reshape(-1, F, 1, 1)
        y = y.repeat(G, 1, H, W)
        x = torch.cat([x, y], dim=1)
        return x

    def extra_repr(self):
        return f"group_size={self.group_size}, num_channels={self.num_channels:d}"


@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        cmap_dim,
        resolution,
        img_channels,
        architecture="resnet",
        mbstd_group_size=4,
        mbstd_num_channels=1,
        activation="lrelu",
        conv_clamp=None,
    ):
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == "skip":
            self.fromrgb = Conv2dLayer(
                img_channels, in_channels, kernel_size=1, activation=activation
            )
        self.mbstd = (
            MinibatchStdLayer(
                group_size=mbstd_group_size, num_channels=mbstd_num_channels
            )
            if mbstd_num_channels > 0
            else None
        )
        self.conv = Conv2dLayer(
            in_channels + mbstd_num_channels,
            in_channels,
            kernel_size=3,
            activation=activation,
            conv_clamp=conv_clamp,
        )
        self.fc = FullyConnectedLayer(
            in_channels * (resolution**2), in_channels, activation=activation
        )
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
        _ = force_fp32
        dtype = torch.float32
        memory_format = torch.contiguous_format

        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == "skip":
            misc.assert_shape(
                img, [None, self.img_channels, self.resolution, self.resolution]
            )
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

    def extra_repr(self):
        return f"resolution={self.resolution:d}, architecture={self.architecture:s}"


@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(
        self,
        c_dim,
        img_resolution,
        img_channels,
        architecture="resnet",
        channel_base=32768,
        channel_max=512,
        num_fp16_res=4,
        conv_clamp=256,
        cmap_dim=None,
        block_kwargs={},
        mapping_kwargs={},
        epilogue_kwargs={},
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2**i for i in range(self.img_resolution_log2, 2, -1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max)
            for res in self.block_resolutions + [4]
        }
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(
            img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp
        )
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = res >= fp16_resolution
            block = DiscriminatorBlock(
                in_channels,
                tmp_channels,
                out_channels,
                resolution=res,
                first_layer_idx=cur_layer_idx,
                use_fp16=use_fp16,
                **block_kwargs,
                **common_kwargs,
            )
            setattr(self, f"b{res}", block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(
                z_dim=0,
                c_dim=c_dim,
                w_dim=cmap_dim,
                num_ws=None,
                w_avg_beta=None,
                **mapping_kwargs,
            )
        self.b4 = DiscriminatorEpilogue(
            channels_dict[4],
            cmap_dim=0,
            resolution=4,
            **epilogue_kwargs,
            **common_kwargs,
        )
        self.convert0 = Conv2dLayer(
            in_channels=1, out_channels=64, kernel_size=3, activation="lrelu"
        )
        self.seg1 = torch.nn.Sequential(
            Conv2dLayer(in_channels=64, out_channels=64, kernel_size=5),
            torch.nn.LeakyReLU(),
            Conv2dLayer(in_channels=64, out_channels=64, kernel_size=5),
            torch.nn.LeakyReLU(),
        )
        self.convert1 = Conv2dLayer(
            in_channels=64, out_channels=128, kernel_size=1, activation="lrelu"
        )
        self.seg2 = torch.nn.Sequential(
            Conv2dLayer(in_channels=128, out_channels=128, kernel_size=5),
            torch.nn.LeakyReLU(),
            Conv2dLayer(in_channels=128, out_channels=128, kernel_size=5),
            torch.nn.LeakyReLU(),
        )

        self.convert2 = Conv2dLayer(
            in_channels=128, out_channels=64, kernel_size=1, activation="lrelu"
        )
        self.seg3 = torch.nn.Sequential(
            Conv2dLayer(in_channels=64, out_channels=64, kernel_size=5),
            torch.nn.LeakyReLU(),
            Conv2dLayer(in_channels=64, out_channels=64, kernel_size=5),
            torch.nn.LeakyReLU(),
        )

        self.seg_out = torch.nn.Sequential(
            Conv2dLayer(in_channels=64, out_channels=64, kernel_size=5),
            torch.nn.LeakyReLU(),
            Conv2dLayer(in_channels=64, out_channels=1, kernel_size=1),
        )

        self.clz = torch.nn.Sequential(
            Conv2dLayer(in_channels=512, out_channels=512, kernel_size=3),
            torch.nn.LeakyReLU(),
            Conv2dLayer(in_channels=512, out_channels=512, kernel_size=3),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(1),
            FullyConnectedLayer(8192, 512, activation="lrelu"),
            FullyConnectedLayer(512, 1),
        )

    def transfer(self, n_classes):
        self.clz = torch.nn.Sequential(
            Conv2dLayer(in_channels=512, out_channels=512, kernel_size=3),
            torch.nn.LeakyReLU(),
            Conv2dLayer(in_channels=512, out_channels=512, kernel_size=3),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(1),
            FullyConnectedLayer(8192, 512, activation="lrelu"),
            FullyConnectedLayer(512, n_classes),
        )

    def forward(
        self, img, update_emas=False, return_features_only=False, **block_kwargs
    ):
        img_norm = (img - img.mean()) / img.std()
        seg = self.convert0(img_norm)
        sc = seg
        seg = self.seg1(seg)
        seg = seg + sc

        seg = self.convert1(seg)
        sc = seg
        seg = self.seg2(seg)
        seg = seg + sc

        seg = self.convert2(seg)
        sc = seg
        seg = self.seg3(seg)
        seg = seg + sc

        seg = self.seg_out(seg)

        _ = update_emas
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f"b{res}")
            x, img = block(x, img, **block_kwargs)
        if return_features_only:
            return x
        clz = self.clz(x)
        x = self.b4(x, img, None)
        return x, seg, clz

    def extra_repr(self):
        return f"c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}"
