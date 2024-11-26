from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='KNN_CUDA',
    version = "0.2",
    ext_modules=[
        CUDAExtension('KNN_CUDA', [
            "/".join(__file__.split('/')[:-1] + ['knn_cuda.cpp']),
            "/".join(__file__.split('/')[:-1] + ['knn.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

