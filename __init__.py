try:
    from .version import __version__
except ModuleNotFoundError:
    pass

from .utils import *
from .GUI import *
from .ext import *