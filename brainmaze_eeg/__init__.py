try:
    from setuptools_scm import get_version
    __version__ = get_version()
except LookupError:
    __version__ = 'dev'  # Fallback version, adjust as appropriate

