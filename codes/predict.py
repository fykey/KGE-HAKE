import os
import json
import logging
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader

from models import KGEModel, ModE, HAKE

from data import TrainDataset, BatchType, ModeType, DataReader
from data import BidirectionalOneShotIterator

# kge_model = torch.load('models/ROBOKG/checkpoint')
print(os.getcwd())
entity_npy = np.load('./models/HAKE_ROBOKG_0/entity_embedding.npy')
relatoin_npy = np.load('./models/HAKE_ROBOKG_0/relation_embedding.npy')
#C:\Users\fuyuki\Desktop\dev\KGE-HAKE\models
config = json.load(open('./models/HAKE_ROBOKG_0/config.json'))

#print()
print(config["hidden_dim"])
the_model = HAKE(len(entity_npy), len(relatoin_npy), config["hidden_dim"], config["gamma"], config["modulus_weight"], config["phase_weight"])
checkpoint = torch.load('./models/HAKE_ROBOKG_0/checkpoint')
the_model.load_state_dict(checkpoint["model_state_dict"])

print(the_model("bottle_1"))