# setup.py
from setuptools import setup, find_packages

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    license="GPLv3",
    name="wsqr-cn",
    version="0.1.7",
    author="twfb",
    author_email="",
    description="Advanced QR Code Decoder Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/twfb/wsqr",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "typer",
        "rich",
        "opencv-python-headless",
        "numpy",
        "psutil",
        "pyzbar",
        "zxing-cpp",
        "qreader",
        "requests",
        "dns_client",
    ],
    entry_points={
        "console_scripts": [
            "wsqr = wsqr_cn.main:app",
        ],
    },
    python_requires=">=3.7",
)
