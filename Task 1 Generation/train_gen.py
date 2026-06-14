import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_gen import MusicGenDataset, collate_gen
from model_gen import build_gen_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MusicGenDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_gen)
model = build_gen_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0.0
    batch_count = 0
    for batch in loader:
        if batch is None:
            continue
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        key_padding_mask = batch['key_padding_mask'].to(device)
        
        output = model(input_ids, key_padding_mask=key_padding_mask)
        loss = criterion(output.view(-1, 130), target_ids.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "best_gen_model.pth")