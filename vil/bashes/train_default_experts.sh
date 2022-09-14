
CPUS=0-20
for ENV in kitchen_ldoor_close-v3 kitchen_micro_close-v3 kitchen_micro_open-v3 
do
taskset -c $CPUS python hydra_mjrl_launcher.py env_name=$ENV job_name=Default \
env_kwargs.sample_layout=False env_kwargs.sample_appliance=False 
done

CPUS=41-60
for ENV in kitchen_rdoor_open-v3 kitchen_rdoor_close-v3  
do
taskset -c $CPUS python hydra_mjrl_launcher.py env_name=$ENV job_name=Default \
env_kwargs.sample_layout=False env_kwargs.sample_appliance=False 
done

CPUS=60-80
for ENV in kitchen_sdoor_close-v3 kitchen_sdoor_open-v3 
do
taskset -c $CPUS python hydra_mjrl_launcher.py env_name=$ENV job_name=Default \
env_kwargs.sample_layout=False env_kwargs.sample_appliance=False 
done

CPUS=67-120
for ENV in kitchen_ldoor_open-v3 kitchen_knob1_on-v3 kitchen_knob1_off-v3 
do
taskset -c $CPUS python hydra_mjrl_launcher.py env_name=$ENV job_name=Default \
env_kwargs.sample_layout=False env_kwargs.sample_appliance=False 
done

CPUS=200-256
for ENV in kitchen_knob3_on-v3 kitchen_knob3_off-v3 #kitchen_knob2_on-v3 kitchen_knob2_off-v3 #
do
taskset -c $CPUS python hydra_mjrl_launcher.py env_name=$ENV job_name=Default \
env_kwargs.sample_layout=False env_kwargs.sample_appliance=False 
done

CPUS=0-60
for ENV in kitchen_ldoor_close-v3  
do
taskset -c $CPUS python hydra_mjrl_launcher.py env_name=$ENV job_name=Default \
env_kwargs.sample_layout=False env_kwargs.sample_appliance=False 
done