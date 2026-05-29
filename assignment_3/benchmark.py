import torch
import torch.nn as nn
import time
import numpy as np

# 1. Original Complex Model (CNN + LSTM)
class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 64, 9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 6)
    def forward(self, x):
        x = self.conv(x).permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.fc(torch.cat((h[-2], h[-1]), dim=1))

# 2. Simplified Model (Pure CNN)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 64, 9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, 6)
    def forward(self, x):
        x = self.conv(x).squeeze(-1)
        return self.fc(x)

def benchmark(model, name, batch_size=64, num_batches=20):
    model.train()
    # Simulated data: (Batch, Channels, Length)
    data = torch.randn(batch_size, 6, 300)
    target = torch.randint(0, 6, (batch_size,))
    criterion = nn.CrossEntropyLoss()
    
    # Warmup
    model(data)
    
    start_time = time.time()
    for _ in range(num_batches):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
    end_time = time.time()
    
    avg_time_per_batch = (end_time - start_time) / num_batches
    return avg_time_per_batch

if __name__ == "__main__":
    print("--- HAR Performance Benchmark ---")
    print("Simulating training speed on your CPU...\n")
    
    complex_model = ComplexModel()
    simple_model = SimpleModel()
    
    t_complex = benchmark(complex_model, "CNN + LSTM")
    t_simple = benchmark(simple_model, "Pure CNN")
    
    # Estimate total training time for one epoch (~140 batches of size 64)
    num_batches_per_epoch = 11020 // 64
    
    print(f"Modell A (CNN + LSTM):")
    print(f"  Sekunden pro Batch: {t_complex:.4f}s")
    print(f"  Geschätzte Zeit pro Epoche: {(t_complex * num_batches_per_epoch)/60:.2f} min")
    
    print(f"\nModell B (Nur CNN):")
    print(f"  Sekunden pro Batch: {t_simple:.4f}s")
    print(f"  Geschätzte Zeit pro Epoche: {(t_simple * num_batches_per_epoch)/60:.2f} min")
    
    improvement = (t_complex - t_simple) / t_complex * 100
    print(f"\nDas einfache Modell ist ca. {improvement:.1f}% schneller.")
    
    if (t_simple * num_batches_per_epoch) > 300:
        print("\nWARNUNG: Selbst das einfache Modell dauert >5 Min pro Epoche.")
        print("Empfehlung: Vielleicht auf eine nicht-Deep-Learning Methode (Random Forest) wechseln.")
    else:
        print("\nDas einfache Modell scheint lokal machbar zu sein.")
