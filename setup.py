from setuptools import find_packages, setup

setup(
    name="adm-diffusers",
    version="0.1.0",
    description="Diffusers-style ADM implementation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy", "torch", "tqdm", "Pillow"],
)
