
#!/bin/sh

script_path=`dirname $0`
script_realpath=`realpath $script_path`
echo "script path = "$script_realpath
top_dir_path=$script_path"/.."
top_dir_realpath=`realpath $top_dir_path`
echo "top dir = "$top_dir_realpath

# create mlir build path
cd ${top_dir_realpath}

# uninstall bad installations
yes|pip3 uninstall cancer
yes|pip3 uninstall pymlir

# invoke setup.py to build wheel installation pkg
cat install_cache.txt | xargs rm -rf
python setup.py install --record install_cache.txt 

# install pymlir
cd external/pymlir
pip3 install .
cd -

# setting up the installation of iree backend
iree_build_dir="${top_dir_realpath}/iree_build/"
export PYTHONPATH=${iree_build_dir}bindings/python:${iree_build_dir}compiler-api/python_package:$PYTHONPATH
