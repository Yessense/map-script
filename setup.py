from setuptools import setup


with open(file='requirements.txt', mode='r') as requirements_handle:
    requirements = requirements_handle.read().splitlines()

with open(file="README.md", mode="r") as readme_handle:
    long_description = readme_handle.read()


setup(
    name='map-script',

    author='Alexander Korchemnyi',
    author_email='akorchemnyj@gmail.com',

    version='0.0.1',

    description='Extract script from text/texts',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/Yessense/map-script',


    packages=["map-script", "map-script.sign", "map-script.preprocessing",
                "map-script.visualization"],
    package_dir={'': 'src'},

    install_requires=requirements,
    include_package_data=True,

    python_requires='>=3.8',

)
