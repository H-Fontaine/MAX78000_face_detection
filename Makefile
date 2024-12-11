ARGS= $(filter-out $@,$(MAKECMDGOALS))

DATASET_FILE=dataset.py
DATASET_NAME=face_classification
DATASET_PATH=datasets/classification
MODEL_FILE=model.py
MODEL_NAME=facenet

LEARNING_RATE=0.001
NB_EPOCHS=10
BATCH_SIZE=40
OPTIMIZER=adam

QAT_POLICY=quantization.yaml
SCHEDULER_POLICY=schedule.yaml


train :
	cd ai8x-training && \
	. .venv/bin/activate && \
	python train.py \
		--lr $(LEARNING_RATE) \
		--optimizer $(OPTIMIZER) \
		--epochs $(NB_EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--compress policies/$(SCHEDULER_POLICY) \
		--qat-policy policies/$(QAT_POLICY) \
		--model $(MODEL_NAME) \
		--dataset $(DATASET_NAME) \
		--data ../$(DATASET_PATH) \
		--confusion \
		--deterministic \
		--param-hist --pr-curves --embedding --device MAX78000 $(ARGS)

QAT_OUT=$(MODEL_NAME)_trained-q.pth.tar
quantize :
	cd ai8x-synthesis && \
	. .venv/bin/activate && \
	python quantize.py ../ai8x-training/latest_log_dir/qat_best.pth.tar trained/$(QAT_OUT) \
		--device MAX78000 -v $(ARGS)

evaluate :
	cd ai8x-training && \
	. .venv/bin/activate && \
	python train.py \
		--model $(MODEL_NAME) \
		--dataset $(DATASET_NAME) \
		--data ../$(DATASET_PATH) \
		--exp-load-weights-from ../ai8x-synthesis/trained/$(QAT_OUT) \
		--device MAX78000 \
		--confusion --evaluate $(ARGS)

links :
	ln -f -s $(CURDIR)/$(MODEL_FILE) ai8x-training/models/$(MODEL_FILE)
	ln -f -s $(CURDIR)/$(DATASET_FILE) ai8x-training/datasets/$(DATASET_FILE)
	ln -f -s $(CURDIR)/$(QAT_POLICY) ai8x-training/policies/$(QAT_POLICY)
	ln -f -s $(CURDIR)/$(SCHEDULER_POLICY) ai8x-training/policies/$(SCHEDULER_POLICY)