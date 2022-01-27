
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Union, Tuple, List

import numpy as np
import torch, pdb
from monai.config import KeysCollection
from monai.transforms.compose import Transform, Randomizable, MapTransform
from monai.networks.layers import GaussianFilter

from monai.transforms.utils import (
    create_grid,
    create_rotate,
    create_scale,
    create_shear,
    create_translate,
)
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    issequenceiterable,
)

RandRange = Optional[Union[Sequence[Union[Tuple[float, float], float]], float]]
GridSampleModeSequence = Union[Sequence[Union[GridSampleMode, str]], GridSampleMode, str]
GridSamplePadModeSequence = Union[Sequence[Union[GridSamplePadMode, str]], GridSamplePadMode, str]
InterpolateModeSequence = Union[Sequence[Union[InterpolateMode, str]], InterpolateMode, str]
NumpyPadModeSequence = Union[Sequence[Union[NumpyPadMode, str]], NumpyPadMode, str]

print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

def totensor(x, d = None):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif x is None:
        return None
    else:
        x = torch.as_tensor(x)
    if d: return x.to(d)
    else: return x

class AffineGridGPU(Transform):
    """
    Affine transforms on the coordinates.

    Args:
        rotate_params: angle range in radians. rotate_params[0] with be used to generate the 1st rotation
            parameter from `uniform[-rotate_params[0], rotate_params[0])`. Similarly, `rotate_params[1]` and
            `rotate_params[2]` are used in 3D affine for the range of 2nd and 3rd axes.
        shear_params: shear_params[0] with be used to generate the 1st shearing parameter from
            `uniform[-shear_params[0], shear_params[0])`. Similarly, `shear_params[1]` to
            `shear_params[N]` controls the range of the uniform distribution used to generate the 2nd to
            N-th parameter.
        translate_params : translate_params[0] with be used to generate the 1st shift parameter from
            `uniform[-translate_params[0], translate_params[0])`. Similarly, `translate_params[1]`
            to `translate_params[N]` controls the range of the uniform distribution used to generate
            the 2nd to N-th parameter.
        scale_params: scale_params[0] with be used to generate the 1st scaling factor from
            `uniform[-scale_params[0], scale_params[0]) + 1.0`. Similarly, `scale_params[1]` to
            `scale_params[N]` controls the range of the uniform distribution used to generate the 2nd to
            N-th parameter.
        as_tensor_output: whether to output tensor instead of numpy array.
            defaults to True.
        device: device to store the output grid data.

    """

    def __init__(
        self,
        rotate_params: Optional[Union[Sequence[float], float]] = None,
        shear_params: Optional[Union[Sequence[float], float]] = None,
        translate_params: Optional[Union[Sequence[float], float]] = None,
        scale_params: Optional[Union[Sequence[float], float]] = None,
        # as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
        verbose = False,
    ) -> None:
        self.rotate_params = rotate_params
        self.shear_params = shear_params
        self.translate_params = translate_params
        self.scale_params = scale_params
        self.verbose = verbose
        # self.as_tensor_output = as_tensor_output
        self.device = device

    def __call__(
        self, spatial_size: Optional[Sequence[int]] = None, grid: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.

        Raises:
            ValueError: When ``grid=None`` and ``spatial_size=None``. Incompatible values.

        """
        if grid is None:
            if spatial_size is not None:
                grid = create_grid(spatial_size)
            else:
                raise ValueError("Incompatible values: grid=None and spatial_size=None.")
        
        device = getattr(grid, 'device') or self.device
        spatial_dims = len(grid.shape) - 1

        affine = np.eye(spatial_dims + 1)
        if self.rotate_params:
            ro_af = create_rotate(spatial_dims, self.rotate_params)
            if self.verbose: print(f'rotate affine {self.rotate_params} \n', ro_af)
            affine = affine @ ro_af
        if self.shear_params:
            affine = affine @ create_shear(spatial_dims, self.shear_params)
        if self.translate_params:
            trans_af = create_translate(spatial_dims, self.translate_params)
            if self.verbose: print(f'translate affine {self.translate_params} \n',  trans_af)
            affine = affine @ trans_af
        if self.scale_params:
            scale_af = create_scale(spatial_dims, self.scale_params)
            if self.verbose: print(f'scale afine {self.scale_params} \n',scale_af)
            affine = affine @ scale_af
        # affine = torch.as_tensor(np.ascontiguousarray(affine), device=self.device)
        # grid = torch.tensor(grid) if not isinstance(grid, torch.Tensor) else grid.detach().clone()
        grid = totensor(grid, device)
        affine = totensor(affine, device).to(grid.dtype)
        # print_tensor('\tgrid', grid)
        if self.verbose: print('\tfinal affine\n', affine)
        grid = (affine @ grid.reshape((grid.shape[0], -1))).reshape([-1] + list(grid.shape[1:]))
        return grid


class RandAffineGridGPU(Randomizable, Transform):
    """
    Generate randomised affine grid.
    """

    def __init__(
        self,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        device = None,
    ) -> None:
        """
        Args:
            rotate_range: angle range in radians. If element `i` is iterable, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the ith dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used. This can
                be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be in range
                `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]` for dim0
                and nothing for the remaining dimensions.
            shear_range: shear_range with format matching `rotate_range`.
            translate_range: translate_range with format matching `rotate_range`.
            scale_range: scaling_range with format matching `rotate_range`. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1).
            as_tensor_output: whether to output tensor instead of numpy array.
                defaults to True.
            device: device to store the output grid data.

        See also:
            - :py:meth:`monai.transforms.utils.create_rotate`
            - :py:meth:`monai.transforms.utils.create_shear`
            - :py:meth:`monai.transforms.utils.create_translate`
            - :py:meth:`monai.transforms.utils.create_scale`
        """
        self.rotate_range = ensure_tuple(rotate_range)
        self.shear_range = ensure_tuple(shear_range)
        self.translate_range = ensure_tuple(translate_range)
        self.scale_range = ensure_tuple(scale_range)

        self.rotate_params: Optional[List[float]] = None
        self.shear_params: Optional[List[float]] = None
        self.translate_params: Optional[List[float]] = None
        self.scale_params: Optional[List[float]] = None

        # self.as_tensor_output = as_tensor_output
        self.device = device

    def _get_rand_param(self, param_range, add_scalar: float = 0.0):
        out_param = []
        for f in param_range:
            if issequenceiterable(f):
                if len(f) != 2:
                    raise ValueError("If giving range as [min,max], should only have two elements per dim.")
                out_param.append(self.R.uniform(f[0], f[1]) + add_scalar)
            elif f is not None:
                out_param.append(self.R.uniform(-f, f) + add_scalar)
        return out_param

    def randomize(self, data: Optional[Any] = None) -> None:
        self.rotate_params = self._get_rand_param(self.rotate_range)
        self.shear_params = self._get_rand_param(self.shear_range)
        self.translate_params = self._get_rand_param(self.translate_range)
        self.scale_params = self._get_rand_param(self.scale_range, 1.0)

    def __call__(
        self, spatial_size: Optional[Sequence[int]] = None, grid: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.

        Returns:
            a 2D (3xHxW) or 3D (4xHxWxD) grid.
        """
        self.randomize()
        affine_grid = AffineGridGPU(
            rotate_params=self.rotate_params,
            shear_params=self.shear_params,
            translate_params=self.translate_params,
            scale_params=self.scale_params,
            # as_tensor_output=self.as_tensor_output,
            device=grid.device,
        )
        return affine_grid(spatial_size, grid)

class ResampleGPU(Transform):
    def __init__(
        self,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
        verbose = False
    ) -> None:
        """
        computes output image using values from `img`, locations from `grid` using pytorch.
        supports spatially 2D or 3D (num_channels, H, W[, D]).

        Args:
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: whether to return a torch tensor. Defaults to False.
            device: device on which the tensor will be allocated.
        """
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.as_tensor_output = as_tensor_output
        self.device = device
        self.verbose = verbose

    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        grid: Optional[Union[np.ndarray, torch.Tensor]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            grid: shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """

        assert isinstance(img, torch.Tensor)
        assert isinstance(grid, torch.Tensor)
        grid = torch.tensor(grid) if not isinstance(grid, torch.Tensor) else grid.detach().clone()
        has_batch_dim = img.dim() == 5
        img_raw_shape = img.shape[-3:]
        for i, size_i in enumerate(img_raw_shape):
            grid[i] = 2.0 * grid[i] / (size_i - 1.0)
        
        if self.verbose: print_tensor(f'grid norm 3c {img_raw_shape}', grid[:3])
        if self.verbose: print_tensor('grid norm 4d', grid[-1:])
        # grid = grid[:-1] / grid[-1:] 

        index_ordering: List[int] = list(range(img.ndimension() - (3 if has_batch_dim else 2) , -1, -1))
        if self.verbose: print(index_ordering) # 2, 1, 0
        grid = grid[index_ordering] # on 0th dim, reorder the xyz > zyx
        grid = grid.permute(list(range(grid.ndimension()))[1:] + [0]) # 0, 1, 2, 3 > 1, 2, 3, 0
        
        if self.verbose: print_tensor('grid norm permute', grid)
        if self.verbose: print(f'mode {self.mode.value} pad {self.padding_mode.value}')
        if has_batch_dim:
            num_batch = img.shape[0]
            grid_shape = grid.shape
            grid = grid.unsqueeze(0).expand(num_batch, *grid_shape)
        else:
            img, grid = img.unsqueeze(0), grid.unsqueeze(0)
        
        out = torch.nn.functional.grid_sample(
            img.float(),
            grid.float(),
            mode=self.mode.value if mode is None else GridSampleMode(mode).value,
            padding_mode=self.padding_mode.value if padding_mode is None else GridSamplePadMode(padding_mode).value,
            align_corners=True)
        
        if self.verbose: print_tensor('input', img)
        if self.verbose: print_tensor('output', out)

        if not has_batch_dim: out = out[0]
        return out

class Rand3DElasticGPUd(Randomizable, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rand3DElastic`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        sigma_range: Tuple[float, float],
        magnitude_range: Tuple[float, float],
        spatial_size: Optional[Union[Tuple[int, int, int], int]] = None,
        prob: float = 0.1,
        rotate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        shear_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        translate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        scale_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadModeSequence = GridSamplePadMode.REFLECTION,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
        verbose = False, 
        meta_key = 'img_meta_dict',
    ) -> None: 
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            sigma_range: a Gaussian kernel with standard deviation sampled from
                ``uniform[sigma_range[0], sigma_range[1])`` will be used to smooth the random offset grid.
            magnitude_range: the random offsets on the grid will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            spatial_size: specifying output image spatial size [h, w, d].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if the components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, 32, -1)` will be adapted
                to `(32, 32, 64)` if the third spatial dimension size of img is `64`.
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            rotate_range: angle range in radians. If element `i` is iterable, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the ith dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used. This can
                be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be in range
                `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]` for dim0
                and nothing for the remaining dimensions.
            shear_range: shear_range with format matching `rotate_range`.
            translate_range: translate_range with format matching `rotate_range`.
            scale_range: scaling_range with format matching `rotate_range`. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1).
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        super().__init__(keys)
        # rotate_angle * np.pi / 180.0
        rotate_range = [a * np.pi / 180.0 for a in rotate_range]
        
        self.rand_affine_grid = RandAffineGridGPU(rotate_range, shear_range, translate_range, scale_range, device)
        self.resampler = ResampleGPU(as_tensor_output=as_tensor_output, device=device)

        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range
        self.spatial_size = spatial_size
        # self.mode: GridSampleMode = GridSampleMode(mode)
        # self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.device = device

        self.prob = prob
        self.do_transform = False
        self.rand_offset = None
        self.magnitude = 1.0
        self.sigma = 1.0
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        # self.grid_tensor = create_grid_torch(spatial_size=spatial_size).half()
        self.grid_tensor = totensor(create_grid(spatial_size)).float() #.half()
        self.verbose = verbose
        self.meta_key = meta_key

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ):
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, grid_size: Sequence[int]) -> None:
        self.do_transform = self.R.rand() < self.prob
        # if self.do_transform:
            # self.rand_offset = self.R.uniform(-1.0, 1.0, [3] + list(grid_size)).astype(np.float32)
        self.unidist = torch.distributions.uniform.Uniform(-1.0, 1.0)
        self.magnitude = self.R.uniform(self.magnitude_range[0], self.magnitude_range[1])
        self.sigma = self.R.uniform(self.sigma_range[0], self.sigma_range[1])
        self.rand_affine_grid.randomize()

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        sp_size = fall_back_tuple(self.spatial_size, data[self.keys[0]].shape[-3:])
        self.randomize(grid_size=sp_size)

        device = d[self.keys[0]].device
        dtype = torch.float if device.type == 'cpu' else torch.half
        if self.verbose: print(f'Device:{device} UseDtype:{dtype}')
        key2dtype = {k : d[k].dtype for k in self.keys}
        # print('ELASTIC', device)
        if self.do_transform:
            # with Timer(print_tmpl='\tElasticTrans {:.3f} seconds'):
            self.grid_tensor = self.grid_tensor.to(device)
            grid4d = self.grid_tensor.clone().to(dtype)
            sigma = torch.tensor(self.sigma, device = device).to(dtype)
            gaussian = GaussianFilter(spatial_dims=3, sigma=sigma, truncated=3.0).to(device) # larger sigma, more flattening curve
            offset4d = self.unidist.rsample([3] + list(sp_size)).to(dtype).to(device).unsqueeze(0)  # (-1.0, 1.0)
            offset_gaus = gaussian(offset4d)[0] # -0.02, 0.02
            grid4d[:3] = grid4d[:3] + offset_gaus * self.magnitude # jitter 1 to 2 depend on the magnitude

            if self.verbose: print_tensor('[RElast] initial grid', self.grid_tensor)
            if self.verbose: print_tensor('[RElast]offset uniform', offset4d)
            if self.verbose: print_tensor(f'[RElast]offset gaussian sigma:{self.sigma} mag:{self.magnitude}', offset_gaus)
            if self.verbose: print_tensor('[RElast]offset grid', grid4d[:3])
            # pdb.set_trace()
            grid4d = self.rand_affine_grid(grid=grid4d)
            if self.verbose: print_tensor('affine grid', grid4d)
            for idx, key in enumerate(self.keys):
                # with Timer(print_tmpl='\tElasticTransform %s {:.3f} seconds' %key):
                if self.verbose: print_tensor(f'[RElast] {key} prev', d[key])
                d[key] = self.resampler(d[key], grid4d, mode=self.mode[idx], 
                                        padding_mode=self.padding_mode[idx])
                d[key] = d[key].to(key2dtype[key])
                if self.verbose: print_tensor(f'[RElast] {key} after', d[key])
                # print_tensor(f'[RandElastic] {key}', d[key])
                for bi, meta_d in enumerate(d[self.meta_key]): meta_d['patch_shape'] = tuple(d[key].shape[-3:])
        return d


from monai.transforms.spatial.array import Flip,  Spacing, DtypeLike

class Flip_(Transform):
    """
    Reverses the order of elements along the given spatial axis. Preserves shape.
    Uses ``np.flip`` in practice. See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        spatial_axis: spatial axes along which to flip over. Default is None.
    """
    def __init__(
            self, spatial_axis: Optional[Union[Sequence[tuple],
                                               tuple]]) -> None:
        self.spatial_axis = spatial_axis

    def __call__(
        self, img: Union[np.ndarray,
                         torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        flipped = list()
        if isinstance(img, torch.Tensor):
            for channel in img:
                flipped.append(torch.flip(channel, self.spatial_axis))
            return torch.stack(flipped)
        elif isinstance(img, np.ndarray):
            f = Flip(self.spatial_axis)
            return f(img)



class SpacingTTAd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Spacing`.

    This transform assumes the ``data`` dictionary has a key for the input
    data's metadata and contains `affine` field.  The key is formed by ``key_{meta_key_postfix}``.

    After resampling the input array, this transform will write the new affine
    to the `affine` field of metadata which is formed by ``key_{meta_key_postfix}``.

    see also:
        :py:class:`monai.transforms.Spacing`
    """

    def __init__(
        self,
        keys: KeysCollection,
        pixdim: Sequence[float],
        diagonal: bool = False,
        mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadModeSequence = GridSamplePadMode.BORDER,
        align_corners: Union[Sequence[bool], bool] = False,
        dtype: Optional[Union[Sequence[DtypeLike], DtypeLike]] = np.float64,
        meta_key_postfix: str = "meta_dict",
        verbose = False,

    ) -> None:
        """
        Args:
            pixdim: output voxel spacing.
            diagonal: whether to resample the input to have a diagonal affine matrix.
                If True, the input data is resampled to the following affine::

                    np.diag((pixdim_0, pixdim_1, pixdim_2, 1))

                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.

                If False, the axes orientation, orthogonal rotation and
                translations components from the original affine will be
                preserved in the target affine. This option will not flip/swap
                axes against the original ones.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of bool, each element corresponds to a key in ``keys``.
            dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.
                It also can be a sequence of dtypes, each element corresponds to a key in ``keys``.
            meta_key_postfix: use `key_{postfix}` to to fetch the meta data according to the key data,
                default is `meta_dict`, the meta data is a dictionary object.
                For example, to handle key `image`,  read/write affine matrices from the
                metadata `image_meta_dict` dictionary's `affine` field.

        Raises:
            TypeError: When ``meta_key_postfix`` is not a ``str``.

        """
        super().__init__(keys)
        self.spacing_transform = Spacing(pixdim, diagonal=diagonal)
        self.diagonal = diagonal
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_key_postfix = meta_key_postfix
        self.verbose = verbose

    def __call__(
        self, data: Mapping[Union[Hashable, str], Dict[str, np.ndarray]]
    ) -> Dict[Union[Hashable, str], Union[np.ndarray, Dict[str, np.ndarray]]]:
        d: Dict = dict(data)
        new_pixdim = d['img_meta_dict'].get('new_pixdim', None)
        # print('Check new pixdim', new_pixdim)
        if new_pixdim is None: return d
        self.spacing_transform = Spacing(new_pixdim, diagonal=self.diagonal, dtype = np.float32)

        for idx, key in enumerate(self.keys):
            meta_data = d[f"{key}_{self.meta_key_postfix}"]
            # resample array of each corresponding key
            # using affine fetched from d[affine_key]
            old_shape = d[key].shape; old_spacing = [meta_data['affine'][i, i] for i in range(3)]
            d[key], _, new_affine = self.spacing_transform(
                data_array=np.asarray(d[key]),
                affine=meta_data["affine"],
                mode=self.mode[idx],
                padding_mode=self.padding_mode[idx],
                align_corners=self.align_corners[idx],
                dtype=self.dtype[idx],
            )
            new_shape = d[key].shape[-3:]
            if self.verbose: print(f'\tRespacing: from {old_shape} {old_spacing} to {new_shape} {new_pixdim}')
            # set the 'affine' key
            meta_data["affine"] = new_affine
            meta_data['shape_post_resize'] = new_shape
        return d


class FlipTTAd_(MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandFlip_`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        keys: Keys to pick data for transformation.
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
    """
    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 1.0,
        spatial_axis: Optional[Union[Sequence[int], int]] = None,
    ) -> None:
        super().__init__(keys)

        self.spatial_axis = spatial_axis

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        # self.randomize()
        d = dict(data)
        flip = d['img_meta_dict']['flip']
        flip_direction = d['img_meta_dict']['flip_direction']

        if flip_direction == 'diagonal': flip_dims = (0, 1)
        elif flip_direction == 'headfeet': flip_dims = (2, )
        elif flip_direction == 'all3axis': flip_dims = (0, 1, 2)
        else: flip_dims = None
        # print(f'PIPELINE: Flip {flip} Direction {flip_direction} dims {flip_dims}')
        if flip and flip_dims is not None: 
            flipper = Flip_(flip_dims)
            for key in self.keys:
                d[key] = flipper(d[key])
        return d



class RandFlipd_(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandFlip_`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        keys: Keys to pick data for transformation.
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
    """
    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        spatial_axis: Optional[Union[Sequence[int], int]] = None,
    ) -> None:
        super().__init__(keys)
        self.spatial_axis = spatial_axis
        self.prob = prob

        self._do_transform = False
        self.flipper = Flip_(spatial_axis=spatial_axis)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._do_transform = self.R.random_sample() < self.prob

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        self.randomize()
        d = dict(data)
        if not self._do_transform:
            return d
        for key in self.keys:
            d[key] = self.flipper(d[key])
        return d