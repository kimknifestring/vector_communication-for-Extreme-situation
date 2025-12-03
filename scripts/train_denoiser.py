import torch
import torch.nn as nn
import torch.optim as optim
import random

VECTOR_FILE = './data/robot_vectors_only_328d.pt'
INPUT_DIM = 328
HIDDEN_DIM = 128

class ResidualDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        # 피드포워드
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, INPUT_DIM)
        )

    def forward(self, x):
        predicted_noise = self.net(x)
        denoised = x - predicted_noise
        return denoised
    
if __name__ == "__main__":
    try:
        clean_vectors = torch.load(VECTOR_FILE)
        print(f">> 원본 벡터 로드 완료. 크기{clean_vectors.shape}")
    except FileNotFoundError:
        print("!! [Error] 벡터 파일이 나이데스요")
        exit()

    model = ResidualDenoiser()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    print("--- 노이즈 제거기 학습 시작 ---")

    num_epochs = 30000

    for epoch in range(num_epochs):
        r = random.uniform(1.0, 10.0)
        attenuation = 1.0 /(r ** 2)
        weak_signal = clean_vectors * attenuation

        noise = torch.randn_like(clean_vectors)
        noisy_input = weak_signal + noise

        target = weak_signal
        reconstructed = model(noisy_input)
        loss = criterion(reconstructed, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss:{loss.item():.6f} (거리:{r:.1f} 상황)")

    torch.save(model.state_dict(), './data/denoiser_model.pth')
    print("-" * 50)
    print(">> 모든 학습 완료!")
    print(">> 저장된 파일: denoiser_model.pth")