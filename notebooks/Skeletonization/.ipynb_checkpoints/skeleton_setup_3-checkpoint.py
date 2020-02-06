from distutils.core import setup, Extension

module = Extension("calcfication_Module",
                   sources = ["skeleton_module.cpp"],
                   extra_link_args=['-leigen3','-lCGAL','-lgmp','-lboost'])

setup(name="Calcfication",
      version = "1.0",
      description = "This is a package for calcification_Module",
      ext_modules = [module])
