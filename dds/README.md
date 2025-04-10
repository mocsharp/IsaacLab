bash
conda activate env_isaaclab

# Install
./isaaclab.sh -i
pip install rti.connext

# Configure Environment
export RTI_LICENSE_FILE=~/Downloads/rti/rti_license.dat 
export PYTHONPATH=$(pwd):$PYTHONPATH


# Run app
./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py --enable_cameras