from setuptools import setup


with open("README.md", "r") as desc:
    long_description = desc.read()
setup(
    name="detectors_world",
    version="0.0.2",
    description="Detectors World is a Python package created for building Computer Vision applications in easy and handy way.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["detectors_world", "detectors", "__init__"],
    package_dir={"": "src"},
    readme="README.md",
    url="https://github.com/MeqdadDev/detectors-world",
    author="Meqdad Dev",
    author_email="meqdad.darweesh@gmail.com",
    install_requires=["mediapipe>=0.9.0.1", "opencv-python>=4.1.2.30"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Artificial Life",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Utilities",
    ],
)
