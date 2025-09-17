from setuptools import setup, find_packages

setup(
    name="bagel",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "torchvision", 
        "transformers",
        "accelerate",
        "bitsandbytes",
        "omegaconf",
        "wandb",
        "pillow",
        "numpy",
        "safetensors",
    ],
)