echo "#### test-cfneb"
python3 -m pytest gpusampling/gpucfneb/test_gpucfneb.py

echo "#### test-metad"
python3 -m pytest gpusampling/test-metad/test_metad.py

echo "#### test-monitor"
python3 -m pytest gpusampling/test-monitor/test_monitor.py
