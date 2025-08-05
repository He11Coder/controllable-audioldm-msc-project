# Experiment name
EXP = test_exp


# Run TensorBoard to observe training results
.PHONY: run-tboard

run-tboard:
	tensorboard --logdir=hifi-gan_runs


# Run training script
.PHONY: run-train

run-train:
	python3 train.py --exp_name='$(EXP)'


# Delete checkpoints of experiment
.PHONY: delete-pt

delete-pt:
	rm -r /scratch/vb00479/audioset_balanced_22k/checkpoints/$(EXP)