from setuptools import setup, find_packages

setup(
    name="histoseg",
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    # 让版本号由 git tag 推导（setuptools-scm 接管）
    use_scm_version=True,
    setup_requires=["setuptools-scm>=8"],
)
