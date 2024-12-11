SLURM_CONF_LOC=/home/sladmitet/slurm/slurm.conf
RESULTS_DIR=results

WORK=work.sh
CLEAN=cleanup.sh
CLEANUP_LOGS=cleanup_logs
CLEANUP_NODES_FILE=used_nodes.txt




run:
	rm -rf $(RESULTS_DIR)
	mkdir $(RESULTS_DIR)
	export SLURM_CONF=$(SLURM_CONF_LOC) && sbatch $(WORK)

queue:
	export SLURM_CONF=$(SLURM_CONF_LOC) && squeue --Format=jobarrayid:10,state:10,partition:16,reasonlist:18,username:10,tres-alloc:45,timeused:8,command:50

info:
	export SLURM_CONF=$(SLURM_CONF_LOC) && sinfo -Node --partition=gpu.normal --Format nodelist:12,statecompact:7,memory:7,allocmem:10,freemem:10,cpusstate:15,cpusload:10,gresused:100



clean:
	rm -rf $(CLEANUP_LOGS)
	mkdir $(CLEANUP_LOGS)
	export SLURM_CONF=$(SLURM_CONF_LOC) && sbatch $(CLEAN)
	rm -f $(CLEANUP_NODES_FILE)

debug:
	export SLURM_CONF=$(SLURM_CONF_LOC) && srun --time 10 --gres=gpu:1 --nodelist=artongpu01 --pty bash -i
	