import torch
import torch.nn as nn
import torch.optim as optim
import os

# 총 30개의 명령
# 0:이동, 1:회전, 2:제어, 3:조작, 4:센싱, 5:상태, 6:신호, 7:통신
data_labels = torch.tensor([
    0, 0, 0, 0,  # 0~3: 이동
    1, 1, 1, 1,  # 4~7: 회전
    2, 2, 2, 2,  # 8~11: 제어
    3, 3, 3, 3, 3, 3, # 12~17: 조작
    4, 4, 4,     # 18~20: 센싱
    5, 5, 5,     # 21~23: 상태
    0, 0, 0,     # 24~26: 이동
    6, 6,        # 27~28: 신호
    7            # 29: 통신
])

class RobotActionEmbedder(nn.Module):
    def __init__(self, num_actions, embedding_dim, num_categories):
        super(RobotActionEmbedder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_actions, embedding_dim=embedding_dim)
        # 카테고리 분류기
        self.classifier_category = nn.Linear(in_features=embedding_dim, out_features=num_categories)
        # 개별 분류기 
        self.classifier_identity = nn.Linear(in_features=embedding_dim, out_features=num_actions)

    def forward(self, input_id):
        vec = self.embedding(input_id)

        out_category = self.classifier_category(vec)
        out_identity = self.classifier_identity(vec)
        
        return vec, out_category, out_identity

NUM_ACTIONS = 30
EMBEDDING_DIM = 328
NUM_CATEGORIES = 8

model = RobotActionEmbedder(NUM_ACTIONS, EMBEDDING_DIM, NUM_CATEGORIES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

print(f"--- 학습 시작 (Dimension: {EMBEDDING_DIM}) ---")

# category, identity 가중치, 1:1로 설정했음
alpha = 0.5
for epoch in range(150):
    inputs = torch.arange(NUM_ACTIONS)
    labels = data_labels

    vectors, pred_category, pred_identity = model(inputs)

    loss_cat = criterion(pred_category, labels)
    loss_id = criterion(pred_identity, inputs)
    total_loss = (alpha * loss_cat) + ((1 - alpha) * loss_id)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss.item():.4f}")
    if float(f"{total_loss.item():.4f}") == 0:
        print("최적 성능 도달, 학습을 중지함.")
        break

final_vectors = model.embedding.weight.data

print(f"\n--- 벡터 형태 확인: {final_vectors.shape} ---") 

torch.save(model.state_dict(), 'robot_action_model_328d.pth')
print(">> 전체 모델 저장 완료: robot_action_model_328d.pth")

torch.save(final_vectors, 'robot_vectors_only_328d.pt')
print(">> 벡터 데이터 저장 완료: robot_vectors_only_328d.pt")