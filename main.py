import config
import trainers

conf = config.HifiGanConfig()

trainer = trainers.Trainer(conf)
trainer.train()