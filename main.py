import config
import trainers

# Create configuration object
conf = config.HifiGanConfig()

# Run training
trainer = trainers.Trainer(conf)
trainer.train()