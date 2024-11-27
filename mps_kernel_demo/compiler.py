import torch.utils.cpp_extension

compiled_lib = torch.utils.cpp_extension.load(
    name='rms_norm',
    sources=['rms_norm.mm'],
    extra_cflags=['-std=c++17'],
   )
