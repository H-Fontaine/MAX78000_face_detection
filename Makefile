SLURM_CONF_LOC=/home/sladmitet/slurm/slurm.conf
WORK=work.sh
CLEAN=cleanup.sh
CLEANUP_LOGS=cleanup_logs
CLEANUP_NODES_FILE=used_nodes.txt




run:
	export SLURM_CONF=$(SLURM_CONF_LOC) && sbatch $(WORK)

clean:
	rm -rf $(CLEANUP_LOGS)
	mkdir $(CLEANUP_LOGS)
	export SLURM_CONF=$(SLURM_CONF_LOC) && sbatch $(CLEAN)
	rm -f $(CLEANUP_NODES_FILE)

	