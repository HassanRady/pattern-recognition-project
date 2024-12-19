from setuptools import setup

torch_required_packages = ["torch", "torchvision", "torchaudio"]

lightning_required_packages = ["lightning", "tensorboard", *torch_required_packages]

required_packages = [
    "pandas",
    "pyarrow",
    "pydantic",
    "scikit-learn>=1.5.2",
    "pyyaml",
    "joblib",
    "optuna",
    "optuna.integration",
    "xgboost",
    "catboost",
    "lightgbm",
    "scipy",
    "tsfresh",
    *torch_required_packages,
    *lightning_required_packages,
    "pytorch-tabnet"
]
dev_required_packages = [
    "mypy",
    "ruff",
    "setuptools",
]

extras_require = {
    "dev": dev_required_packages,
}

setup(
    name="prp-hassan-rady",
    version="1.0.0",
    description="Pattern Recognition Project",
    author="Hassan Rady",
    license="MIT",
    install_requires=required_packages,
    extras_require=extras_require,
)
