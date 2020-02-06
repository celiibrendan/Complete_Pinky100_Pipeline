from distutils.core import setup, Extension

module = Extension("calcification_Module",
                   sources = ["skeleton_module.cpp"],
                   #extra_compile_args=['-v','-std=c++0x'],
                   #extra_link_args=['-L /usr/include/','-lCGAL','-lgmp','-lmpfr','-lboost_container','-std=c++0x'],
                   extra_link_args=['-lCGAL','-lgmp','-lmpfr'],
                   #library_dirs = ['/usr/include/eigen3'],
                   include_dirs=['/usr/include/eigen3'])

setup(name="Calcfication",
      version = "1.0",
      description = "This is a package for calcification_Module",
      ext_modules = [module])