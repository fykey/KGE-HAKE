import numpy as np
import matplotlib.pyplot as plt
import os


model_path = r"../models/ROBOKG"
embedding_range = 1e-3
head_id = 0
tail_id = 101

entity_embedding = np.load(os.path.join(model_path, r'entity_embedding.npy'))
print(len(entity_embedding))
print(len(entity_embedding[0]))
    
head = entity_embedding[head_id]
tail = entity_embedding[tail_id]
print(len(head))

phase_head, mod_head = np.split(head, 2)
phase_tail, mod_tail = np.split(tail, 2)
print(len(phase_head))

mod_head = np.log(np.abs(mod_head))  * np.sign(mod_head)
mod_tail = np.log(np.abs(mod_tail))  * np.sign(mod_tail)

phase_head = phase_head / embedding_range * np.pi
phase_tail = phase_tail / embedding_range * np.pi

x_head, y_head = mod_head * np.cos(phase_head), mod_head * np.sin(phase_head)
x_tail, y_tail = mod_tail * np.cos(phase_tail), mod_tail * np.sin(phase_tail)

plt.scatter(x_head, y_head)
plt.scatter(x_tail, y_tail)
plt.axis('equal')
plt.show()