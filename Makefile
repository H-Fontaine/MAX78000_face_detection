# Defaults names for the files
CONFIG_FILE=config.yaml
QAT_POLICY=quantization.yaml
SCHEDULER_POLICY=schedule.yaml
MODEL_FILE=model.py

# Compilation variables
MAXIM_PATH=$(HOME)/MaximSDK
PREFIX=arm-none-eabi-
GDB=$(PREFIX)gdb

# Dataset and model names
DATASET=classification
MODEL=facenet_v2

# Training variables
LEARNING_RATE=0.001
NB_EPOCHS=15
BATCH_SIZE=40
OPTIMIZER=adam

# Paths to the files
DATASET_PATH=datasets/$(DATASET)

train : links
	cd ai8x-training && \
	. .venv/bin/activate && \
	python train.py \
		--lr $(LEARNING_RATE) \
		--optimizer $(OPTIMIZER) \
		--epochs $(NB_EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--compress policies/$(SCHEDULER_POLICY) \
		--qat-policy policies/$(QAT_POLICY) \
		--model $(MODEL) \
		--dataset $(DATASET) \
		--data ../$(DATASET_PATH) \
		--confusion \
		--deterministic \
		--param-hist --pr-curves --embedding --device MAX78000 $(ARGS)

QAT_OUT=$(MODEL)_trained-q.pth.tar
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
		--model $(MODEL) \
		--dataset $(DATASET) \
		--data ../$(DATASET_PATH) \
		--exp-load-weights-from ../ai8x-synthesis/trained/$(QAT_OUT) \
		--device MAX78000 \
		--save-sample 0 \
		--8-bit-mode \
		--confusion --evaluate $(ARGS)

OUT_SYNTHESIS=synthed_nets
synthesize :
	rm -f ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL)/main.c
	cd ai8x-synthesis && \
	. .venv/bin/activate && \
	python ai8xize.py \
		--test-dir $(OUT_SYNTHESIS) \
		--prefix $(MODEL) \
		--checkpoint-file trained/$(QAT_OUT) \
		--config-file networks/$(CONFIG_FILE) \
		--sample-input ../ai8x-training/sample_$(DATASET).npy \
		--softmax \
		--compact-data \
		--mexpress --timer 0 --display-checkpoint --overwrite --verbose --device MAX78000 $(ARGS)
	cd ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL) && \
	tail -n 100 main.c | sed -n '/\/\*/,/\*\//{/\/\*/d;/\*\//d;p}' > $(CURDIR)/models/$(MODEL)/ops.txt

camera: clean
	ln -f -s $(CURDIR)/camera/main.c ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL)/main.c
	ln -f -s $(CURDIR)/camera/utils.c ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL)/utils.c
	ln -f -s $(CURDIR)/camera/utils.h ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL)/utils.h
	ln -f -s $(CURDIR)/camera/project.mk ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL)/project.mk


links:
	ln -f -s $(CURDIR)/models/$(MODEL)/$(MODEL_FILE) ai8x-training/models/$(MODEL).py
	ln -f -s $(CURDIR)/models/$(MODEL)/$(QAT_POLICY) ai8x-training/policies/$(QAT_POLICY)
	ln -f -s $(CURDIR)/models/$(MODEL)/$(SCHEDULER_POLICY) ai8x-training/policies/$(SCHEDULER_POLICY)
	ln -f -s $(CURDIR)/models/$(MODEL)/$(CONFIG_FILE) ai8x-synthesis/networks/$(CONFIG_FILE)
	ln -f -s $(CURDIR)/datasets/$(DATASET)_dataset.py ai8x-training/datasets/$(DATASET)_dataset.py

server:
	cd ai8x-synthesis/openocd/ && \
	./run-openocd-maxdap

build:
	cd ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL_NAME) && \
	make MAXIM_PATH=$(MAXIM_PATH)

clean:
	cd ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL_NAME) && \
	make clean MAXIM_PATH=$(MAXIM_PATH)


flash: build
	$(GDB) -x config.gdb ai8x-synthesis/$(OUT_SYNTHESIS)/$(MODEL_NAME)/build/max78000.elf

listen :
	clear
	tio /dev/serial/by-id/usb-ARM_DAPLink_CMSIS-DAP_04441701c0e38ade00000000000000000000000097969906-if01
