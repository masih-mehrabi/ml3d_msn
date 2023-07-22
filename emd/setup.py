from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='emd',
    ext_modules=[
        CUDAExtension('emd', [
            '/content/ml3d_msn/emd/emd.cpp',
            '/content/ml3d_msn/emd/emd_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })