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

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]


# Load entity and relation embeddings
entity_npy = np.load('./models/HAKE_ROBOKG_0/entity_embedding.npy')
relation_npy = np.load('./models/HAKE_ROBOKG_0/relation_embedding.npy')


enitity_path = './data/ROBOKG/entities.dict'
relation_path = './data/ROBOKG/relations.dict'
# ファイルを読み込み、IDとラベルを対応させる辞書を作成
entity_dict = {}
relation_dict = {}
with open(enitity_path, 'r') as file:
    for line in file:
        # 各行をスペースまたはタブで分割
        parts = line.strip().split('\t')
        if len(parts) == 2:
            # IDとラベルを辞書に追加
            entity_dict[int(parts[0])] = parts[1]

with open(relation_path, 'r') as file:
    for line in file:
        # 各行をスペースまたはタブで分割
        parts = line.strip().split('\t')
        if len(parts) == 2:
            # IDとラベルを辞書に追加
            relation_dict[int(parts[0])] = parts[1]
#/TaskInstance/Grasp_Apple_1	_WhichForce
# Load model configuration
config = json.load(open('./models/HAKE_ROBOKG_0/config.json'))

# Create an instance of the HAKE model
the_model = HAKE(len(entity_dict), len(relation_dict), config["hidden_dim"], config["gamma"], config["modulus_weight"], config["phase_weight"])
#summary(model=the_model, input_size=[(1, 1, 1), (1, 1, 1), (1, 1, 1)], batch_size=-1, device='cpu')


head_index = keys = get_keys_from_value(entity_dict, '/TaskCategory/Grasp')
relation_index = keys = get_keys_from_value(relation_dict, '_SpecifyTask')
tail_index = list(range(len(entity_dict)))
#tail_index.remove(head_index[0])

# Prepare input tensors for the model
head_tensor = torch.from_numpy(entity_npy[head_index]).unsqueeze(0)  # Assuming head_tensor has shape [1, hidden_dim * 2]
relation_tensor = torch.from_numpy(relation_npy[relation_index]).unsqueeze(0)  # Assuming relation_tensor has shape [1, hidden_dim * 3]
tail_tensor = torch.from_numpy(entity_npy[tail_index]).unsqueeze(0)  # Assuming tail_tensor has shape [num_entities - 1, hidden_dim * 2]
# Load the trained model weights
checkpoint = torch.load('./models/HAKE_ROBOKG_0/checkpoint_10000')
#checkpoint = torch.load('./checkpoint')
the_model.load_state_dict(checkpoint["model_state_dict"])

# Set the model to evaluation mode
the_model.eval()
score = []
# Perform the prediction
with torch.no_grad():
    # Modify the following line to pass 'None' for the 'tail' parameter
    score = 9-the_model.func(head_tensor, relation_tensor, tail_tensor, BatchType.TAIL_BATCH)
print(score.size())
print(score)
print("minimum score = ", score[0,torch.argmin(score, dim=1)].item())
print(entity_dict[torch.argmin(score, dim=1).item()])


# Convert the predicted tensor to a numpy array
#predicted_tail = predicted_tail.squeeze().cpu().numpy()

# Return the predicted tail entity
#print("Predicted Tail Entity:", predicted_tail)