"""Setup file"""
import setuptools

package_name = "ncpyview"
version = "1.0.0"
author = "Wassim Fkaier"
setuptools.setup(
    name=package_name,
    version=version,
    author=author,
    author_email="wassfk@outlook.com",
    description="Interactive web application",
    long_description="Interactive web application for netcdf file data visualization",
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    package_data={package_name: ["etc/*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy", "streamlit", "xarray", "plotly"],
    entry_points={"console_scripts": ["ncpyview = %s.main:main" % package_name]},
)
