from setuptools import setup, find_packages

setup(
    name="styletss2",
    version="0.1.0",
    packages=find_packages(),
    py_modules=[
        "inference",
        "models",
        "utils",
        "text_utils"
    ],
    install_requires=[
        "SoundFile",
        "torchaudio",
        "munch",
        "torch",
        "pydub",
        "PyYAML",
        "librosa",
        "nltk",
        "matplotlib",
        "accelerate",
        "transformers",
        "einops",
        "einops-exts",
        "tqdm",
        "typing-extensions",
        # VCS dependency (pip will install it)
        "monotonic-align @ git+https://github.com/resemble-ai/monotonic_align.git",
    ]
)
