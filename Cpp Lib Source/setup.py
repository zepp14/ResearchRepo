from distutils.core import setup, Extension

module = Extension("zeppNumLib", sources=["myModule.cpp"],include_dirs=['/usr/local/include/Eigen'])

setup(name="MyPackageName", version='1.0', description='This is Mine',
ext_modules = [module])
