from setuptools import setup, find_packages
#from distutils.extension import Extension
#from Cython.Build import cythonize


def readme():
    with open('README.md') as f:
        return f.read()

requirements = [
    "numpy",
    "scipy",
    "cython",
    "sklearn",
    "mpi4py",

    # For profiling.
    #"cProfile",
    #"LineProfiler",
    #"pstats",
]

setup(
    name="xcsvm",
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
    packages=find_packages(),
    install_requires=requirements,
    # We use pyximport and compile at runtime.
    #ext_modules=Extension(name="*",
    #                      sources=["xcsvm/solvers/cython/*/*.pyx"],
    #                      extra_link_args=['-fopenmp'],
    #                      extra_compile_args=['-O3', '-finline-functions',
    #                                          '-fopenmp']),
    # test_suite='nose.collector',
    # tests_require=['nose']+requirements,
    include_package_data=True,
    zip_safe=False,
)
