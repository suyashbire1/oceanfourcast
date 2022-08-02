from setuptools import setup

setup(
    name='oceanfourcast',
    version='0.1.0',
    description='FourcastNet applied to ocean simulation',
    url='https://github.com/suyashbire1/oceanfourcast',
    author='Suyash Bire',
    author_email='bire@mit.edu',
    license='MIT',
    packages=['oceanfourcast'],
    install_requires=['torch',
                      'numpy',
                      'xarray',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
    ],
)
