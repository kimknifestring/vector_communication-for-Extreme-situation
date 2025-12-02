import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. 기본 설정 ---
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

VECTOR_FILE = 'robot_vectors_only_328d.pt'

CATEGORY_NAMES_LIST = ['이동', '회전', '제어', '조작', '감지', '상태', '신호', '통신']

# 학습에 사용된 정답 라벨 (Centroid 계산용)
# 0~29번 데이터가 어떤 카테고리인지 매핑
data_labels_np = np.array([
    0, 0, 0, 0,  # 0~3: Move
    1, 1, 1, 1,  # 4~7: Rotate
    2, 2, 2, 2,  # 8~11: Control
    3, 3, 3, 3, 3, 3, # 12~17: Manipulate
    4, 4, 4,     # 18~20: Sense
    5, 5, 5,     # 21~23: Status
    0, 0, 0,     # 24~26: Move (뒤에 있는 Move)
    6, 6,        # 27~28: Signal
    7            # 29: Comm
])

try:
    vectors = torch.load(VECTOR_FILE).numpy()
    print(f">> 벡터 로드 성공! Shape: {vectors.shape}")
except FileNotFoundError:
    print(f"!! 오류: '{VECTOR_FILE}' 파일이 없습니다. 학습 코드를 먼저 실행하세요.")
    exit()

# --- 3. 모드 선택 ---
print("-" * 50)
print("1: [Token vs Token] 각 동작 별 코사인 유사도")
print("2: [Token vs Category] 각 동작과 카테고리 유사도")
print("-" * 50)

try:
    control = int(input("모드를 선택하세요 (1 또는 2): "))
except ValueError:
    print("숫자를 입력해주세요.")
    exit()

# --- 4. 시각화 로직 ---
if control == 1:
    # ==========================================
    # 모드 1: 동작 vs 동작 (30x30)
    # ==========================================
    print(">> 1. 토큰 간 코사인 유사도 히트맵 생성 중...")

    sim_matrix = cosine_similarity(vectors)

    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix, 
                cmap='turbo', 
                vmin=0.0, vmax=1.0, 
                annot=False) # 숫자가 너무 많으니 끔 (필요하면 True)

    plt.title('전체 동작(Token) 간 코사인 유사도', fontsize=16, weight='bold')
    plt.xlabel('Action ID (0~29)')
    plt.ylabel('Action ID (0~29)')
    
    print(">> 히트맵 출력")
    plt.show()

elif control == 2:
    # ==========================================
    # 모드 2: 동작 vs 카테고리 (30x8)
    # ==========================================
    print(">> 2. 토큰 vs 카테고리 중심 유사도 분석 중...")

    # (1) 카테고리별 대표 벡터(Centroid) 계산
    centroids = []
    for cat_idx in range(8): # 0~7번 카테고리
        # 해당 카테고리에 속하는 인덱스 찾기
        indices = np.where(data_labels_np == cat_idx)[0]
        # 해당 벡터들의 평균 계산
        cat_centroid = np.mean(vectors[indices], axis=0)
        centroids.append(cat_centroid)
    
    # 리스트를 numpy 배열로 변환 [8, 328]
    centroids = np.array(centroids)

    # (2) 유사도 측정: [30개 동작] vs [8개 카테고리]
    sim_matrix = cosine_similarity(vectors, centroids)

    # (3) 시각화
    plt.figure(figsize=(10, 12))
    sns.heatmap(sim_matrix, 
                cmap='turbo', 
                vmin=0.0, vmax=1.0, 
                annot=True, fmt='.2f',  # 여기는 숫자를 보는 게 중요함
                xticklabels=CATEGORY_NAMES_LIST,
                yticklabels=range(30),
                annot_kws={"size": 9})

    plt.title('각 토큰 별 카테고리 유사도', fontsize=16, weight='bold')
    plt.xlabel('Category Centroids (카테고리 중심)')
    plt.ylabel('Individual Action ID (개별 동작)')
    plt.show()

else:
    print("!! 잘못된 입력입니다. 1 또는 2를 입력하세요.")