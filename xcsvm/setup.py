from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name="xcsvm",
      version="0.1",
      description="Support Vector Machines for Extreme Classfication tasks.",
      long_description=readme(),
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Development Status :: 3 - Alpha",
          "Environment :: Console",
          "Intended Audience :: Science/Research",
          "Operating System :: POSIX :: Linux",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Cython",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ],
      url="http://github.com/albermax/xcsvm",
      author="Maxmilian Alber",
      author_email="albermax@github.com",
      license="MIT",
      packages=["xcsvm"],
      install_requires=[
          "nose",
          "numpy",
      ],
      include_package_data=True,
      zip_safe=False)
