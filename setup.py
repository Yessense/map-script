from setuptools import setup

setup(
    name='script-extract',
    version='0.0.1',
    description='Extract script from text/texts',
    py_modules=["script_extraction", "script_extraction.sign", "script_extraction.preprocessing",
                "script_extraction.visualization"],
    package_Dir={'script-extract': 'src'}
)
