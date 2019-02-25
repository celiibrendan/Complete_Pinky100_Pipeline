from distutils.core import setup, Extension

module = Extension("calcfication_Module_2",
                   sources = ["skeleton_module_2.cpp"],
                   extra_link_args=['-leigen3','-lCGAL','-lgmp','-lboost'])

setup(name="Calcfication",
      version = "1.0",
      description = "This is a package for calcification_Module",
      ext_modules = [module])
