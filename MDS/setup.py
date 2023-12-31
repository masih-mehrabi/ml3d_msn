from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='MDS',
    ext_modules=[
        CUDAExtension('MDS', [
            '/content/ml3d_msn/MDS/MDS_cuda.cu',
            '/content/ml3d_msn/MDS/MDS.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })