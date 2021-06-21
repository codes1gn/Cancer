find ./lib -exec sed -i 's/npcomp/cancer/g' {} \;
find ./lib -exec sed -i 's/NPCOMP/CANCER/g' {} \;
find ./lib -exec sed -i 's/\"cancer\//\"/g' {} \;

find ./include -exec sed -i 's/npcomp/cancer/g' {} \;
find ./include -exec sed -i 's/NPCOMP/CANCER/g' {} \;
find ./include -exec sed -i 's/\"cancer\//\"/g' {} \;

find ./test -exec sed -i 's/npcomp/cancer/g' {} \;
find ./test -exec sed -i 's/-run-mlir/-runner/g' {} \;
