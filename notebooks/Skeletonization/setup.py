from distutils.core import setup, Extension

module = Extension("calcification_Module",
                   sources = ["skeleton_module.cpp"],
                   extra_link_args=['-lCGAL','-lgmp'],
                   library_dirs = ['/usr/local/include/eigen3'],
                   include_dirs=['/usr/local/include/eigen3'])

setup(name="Calcfication",
      version = "1.0",
      description = "This is a package for calcification_Module",
      ext_modules = [module])
