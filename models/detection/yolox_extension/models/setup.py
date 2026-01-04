from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='linesegment_cuda',
    ext_modules=[
        CUDAExtension(
            name='linesegment_cuda',
            sources=[
                'csrc/binding.cpp', 
                'csrc/linesegment.cu'
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)