import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.Linear(10, 2).to(device)
x = torch.randn(4, 10, device=device)
y = torch.randint(0, 2, (4,), device=device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

outputs = model(x)
loss = criterion(outputs, y)
optimizer.zero_grad()
loss.backward()  # If this fails with CUBLAS_STATUS_NOT_INITIALIZED => likely a setup/env issue
optimizer.step()

print("Success: minimal forward/backward works on GPU")