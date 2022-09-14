CPUS=0-48
for ENV in kitchen_sdoor_open-v3 
do
for SEED in {103..113}
do
taskset -c $CPUS python hydra_mjrl_launcher.py env_name=$ENV job_name=Heldout \
env_kwargs.sample_layout=True env_kwargs.sample_appliance=True env_seed=${SEED} \
rl_num_iter=300
done
done


for ENV in kitchen_micro_open-v3 
do
for SEED in {101..110}
do
taskset -c $CPUS python hydra_mjrl_launcher.py env_name=$ENV job_name=Heldout \
env_kwargs.sample_layout=True env_kwargs.sample_appliance=True env_seed=${SEED} \
rl_num_iter=300
done
done

for ENV in kitchen_knob1_on-v3 kitchen_knob1_off-v3 kitchen_knob2_on-v3 kitchen_knob2_off-v3  kitchen_knob3_on-v3 kitchen_knob3_off-v3 kitchen_knob4_on-v3 kitchen_knob4_off-v3
do
for SEED in 101 102 
do 
taskset -c $CPUS python hydra_mjrl_launcher.py env_name=$ENV job_name=Heldout \
env_kwargs.sample_layout=True env_kwargs.sample_appliance=True env_seed=${SEED} \
rl_num_iter=300
done 
done 

ENV=kitchen_micro_close-v3

ENV=kitchen_rdoor_close-v3
for SEED in 103 104 105 106 107 108 109 110 
do 
taskset -c $CPUS python hydra_mjrl_launcher.py env_name=$ENV job_name=Heldout \
env_kwargs.sample_layout=True env_kwargs.sample_appliance=True env_seed=${SEED} \
rl_num_iter=300
done 