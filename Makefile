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

links :
	ln -f -s $(CURDIR)/$(MODEL_FILE) ai8x-training/models/$(MODEL_FILE)
	ln -f -s $(CURDIR)/$(DATASET_FILE) ai8x-training/datasets/$(DATASET_FILE)
	ln -f -s $(CURDIR)/$(QAT_POLICY) ai8x-training/policies/$(QAT_POLICY)
	ln -f -s $(CURDIR)/$(SCHEDULER_POLICY) ai8x-training/policies/$(SCHEDULER_POLICY)