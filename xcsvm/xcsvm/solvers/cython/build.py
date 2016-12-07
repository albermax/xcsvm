
trace = []
if False:
    # needs to be set when using line profiler
    trace = ['-DCYTHON_TRACE=1', '-DCYTHON_TRACE_NOGIL=1']


def make_ext(modname, pyxfilename):
    from distutils.extension import Extension
    return Extension(name=modname,
                     sources=[pyxfilename],
                     extra_link_args=['-fopenmp'],
                     extra_compile_args=['-O3', '-finline-functions',
                                         '-fopenmp'] + trace)
