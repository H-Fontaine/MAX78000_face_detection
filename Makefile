ARGS= $(filter-out $@,$(MAKECMDGOALS))

MAXIM_PATH=$(HOME)/MaximSDK
PREFIX=arm-none-eabi-
GDB=$(PREFIX)gdb

DATASET_FILE=dataset.py
DATASET_NAME=face_classification
DATASET_PATH=datasets/classification
MODEL_FILE=model.py
MODEL_NAME=facenet
CONFIG_FILE=config.yaml
QAT_POLICY=quantization.yaml
SCHEDULER_POLICY=schedule.yaml

LEARNING_RATE=0.001
NB_EPOCHS=10
BATCH_SIZE=40
OPTIMIZER=adam



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
quantize : #quantize the last trained model
	cd ai8x-synthesis && \
	. .venv/bin/activate && \
	LATEST_FOLDER=$$(find ../ai8x-training/logs -type d -exec test -e {}/qat_best.pth.tar \; -print | sort -r | head -n 1) && \
	python quantize.py $$LATEST_FOLDER/qat_best.pth.tar trained/$(QAT_OUT) \
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
		--save-sample 0 \
		--8-bit-mode \
		--confusion --evaluate $(ARGS)

OUT_SYNTHESIS=synthed_nets
synthesize :
	rm -f ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL_NAME)/main.c
	cd ai8x-synthesis && \
	. .venv/bin/activate && \
	python ai8xize.py \
		--test-dir $(OUT_SYNTHESIS) \
		--prefix $(MODEL_NAME) \
		--checkpoint-file trained/$(QAT_OUT) \
		--config-file networks/$(CONFIG_FILE) \
		--sample-input ../ai8x-training/sample_$(DATASET_NAME).npy \
		--softmax \
		--compact-data \
		--mexpress --timer 0 --display-checkpoint --overwrite --verbose --device MAX78000 $(ARGS)

camera::
	ln -f -s $(CURDIR)/camera/main.c ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL_NAME)/main.c
	ln -f -s $(CURDIR)/camera/utils.c ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL_NAME)/utils.c
	ln -f -s $(CURDIR)/camera/utils.h ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL_NAME)/utils.h
	ln -f -s $(CURDIR)/camera/project.mk ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL_NAME)/project.mk


links::
	ln -f -s $(CURDIR)/$(MODEL_FILE) ai8x-training/models/$(MODEL_FILE)
	ln -f -s $(CURDIR)/$(DATASET_FILE) ai8x-training/datasets/$(DATASET_FILE)
	ln -f -s $(CURDIR)/$(QAT_POLICY) ai8x-training/policies/$(QAT_POLICY)
	ln -f -s $(CURDIR)/$(SCHEDULER_POLICY) ai8x-training/policies/$(SCHEDULER_POLICY)
	ln -f -s $(CURDIR)/$(CONFIG_FILE) ai8x-synthesis/networks/$(CONFIG_FILE)

server:
	cd ai8x-synthesis/openocd/ && \
	./run-openocd-maxdap

build:
	cd ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL_NAME) && \
	make MAXIM_PATH=$(MAXIM_PATH)

flash: build
	$(GDB) -x config.gdb ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL_NAME)/build/max78000.elf

listen :
	clear
	tio /dev/serial/by-id/usb-ARM_DAPLink_CMSIS-DAP_04441701c0e38ade00000000000000000000000097969906-if01
