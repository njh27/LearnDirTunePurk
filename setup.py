from distutils.core import setup
import numpy as np



setup(name='LearnDirTunePurk',
      version='1.0',
      description='Analysis code for direction learning with purkinje cell neural recordings.',
      author='Nathan Hall',
      author_email='nathan.hall@duke.edu',
      url='https://',
      packages=['LearnDirTunePurk'],
      include_dirs=[np.get_include()],
     )
