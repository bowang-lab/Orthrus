from pkg_resources import parse_version
from configparser import ConfigParser
import setuptools
from setuptools import Extension
from Cython.Distutils import build_ext
import numpy

# Ensure setuptools is sufficiently up to date
assert parse_version(setuptools.__version__) >= parse_version("36.2")

# Load config from settings.ini
config = ConfigParser(delimiters=["="])
config.read("settings.ini")
cfg = config["DEFAULT"]


# Lazy approach for getting NumPy include path
class get_numpy_include:
    """Returns NumPy's include path with lazy import."""

    def __str__(self):
        import numpy

        return numpy.get_include()


# Example Cython extension module: orthrus._sequence
genome_module = Extension(
    "orthrus._sequence", ["orthrus/_sequence.pyx"], include_dirs=[numpy.get_include()]
)
ext_modules = [genome_module]
cmdclass = {"build_ext": build_ext}

# Required keys in settings.ini
cfg_keys = "version description keywords author author_email".split()
expected = (
    cfg_keys
    + "lib_name user branch license status min_python audience language".split()
)
for key in expected:
    assert key in cfg, f"Missing expected setting: {key}"

# We will map a few fields from config to setup()
setup_cfg = {k: cfg[k] for k in cfg_keys}

# License mapping
licenses = {
    "apache2": (
        "Apache Software License 2.0",
        "OSI Approved :: Apache Software License",
    ),
    "mit": ("MIT License", "OSI Approved :: MIT License"),
    "gpl2": (
        "GNU General Public License v2",
        "OSI Approved :: GNU General Public License v2 (GPLv2)",
    ),
    "gpl3": (
        "GNU General Public License v3",
        "OSI Approved :: GNU General Public License v3 (GPLv3)",
    ),
    "bsd3": ("BSD License", "OSI Approved :: BSD License"),
}

statuses = [
    "1 - Planning",
    "2 - Pre-Alpha",
    "3 - Alpha",
    "4 - Beta",
    "5 - Production/Stable",
    "6 - Mature",
    "7 - Inactive",
]

py_versions = "3.6 3.7 3.8 3.9 3.10".split()
requirements = cfg.get("requirements", "").split()
if cfg.get("pip_requirements"):
    requirements += cfg.get("pip_requirements", "").split()

min_python = cfg["min_python"]
lic = licenses.get(cfg["license"].lower(), (cfg["license"], None))
dev_requirements = (cfg.get("dev_requirements") or "").split()

# Construct the setup call
setuptools.setup(
    name=cfg["lib_name"],
    license=lic[0],
    classifiers=[
        "Development Status :: " + statuses[int(cfg["status"])],
        "Intended Audience :: " + cfg["audience"].title(),
        "Natural Language :: " + cfg["language"].title(),
    ]
    + [
        "Programming Language :: Python :: " + ver
        for ver in py_versions[py_versions.index(min_python) :]
    ]
    + (["License :: " + lic[1]] if lic[1] else []),
    url=cfg["git_url"],
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    dependency_links=cfg.get("dep_links", "").split(),
    python_requires=">=" + cfg["min_python"],
    # If you have a README.md, we can set it as the long description:
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    zip_safe=False,
    # Optional: console entry points, or nbdev integration
    entry_points={
        "console_scripts": cfg.get("console_scripts", "").split(),
        "nbdev": [f"{cfg.get('lib_path')}={cfg.get('lib_path')}._modidx:d"],
    },
    # Unpack the fields from the config into setup()
    **setup_cfg,
)
