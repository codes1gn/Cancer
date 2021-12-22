
script_path=`dirname $0`
script_realpath=`realpath $script_path`
echo "script path = "$script_realpath
top_dir_path=$script_path"/.."
top_dir_realpath=`realpath $top_dir_path`
echo "top dir = "$top_dir_realpath

# TODO change into a inspect code, to avoid recompile before test
# TODO temporarilly ban compile, for fast test
# sh ${top_dir_realpath}/scripts/_build_iree.sh

iree_build_dir="${top_dir_realpath}/iree_build/"

# add python module path
export PYTHONPATH=${iree_build_dir}bindings/python:$PYTHONPATH
export PYTHONPATH=${iree_build_dir}compiler-api/python_package:$PYTHONPATH

# verify the installation
python3 -c "import iree.compiler"
python3 -c "import iree.runtime"

# testing
python backend_tests/iree/test_add_tosa.py
