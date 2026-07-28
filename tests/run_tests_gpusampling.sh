echo "#### test-cfneb"
python3 -m pytest gpusampling/gpucfneb/test_gpucfneb.py

echo "#### test-metad"
python3 -m pytest gpusampling/test-metad/test_metad.py
