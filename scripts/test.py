import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch


from transmitter import sender, channel, receiver, DATA_LABELS

MAX_DISTANCE = 5
TRIALS_PER_STEP = 100 

def run_automated_experiment():
    print("=" * 60)
    print(f"실험 시작")
    print("=" * 60)

    results = []
    
    distances = np.arange(1.0, MAX_DISTANCE, 0.1)
    
    total_steps = len(distances) * TRIALS_PER_STEP
    current_step = 0

    for r in distances:
        for _ in range(TRIALS_PER_STEP):

            target_id = random.randint(0, 29)
            tx_vector = sender(target_id)
            rx_vector = channel(tx_vector, distance_r=r)
            decoded_id, decoded_name, confidence = receiver(rx_vector)
                    
            is_success = (target_id == decoded_id)
            
            results.append({
                "Distance": r,
                "Target_ID": target_id,
                "Decoded_ID": decoded_id,
                "Success": 1 if is_success else 0,
                "Confidence": confidence,
                "Category_Label": DATA_LABELS[target_id]
            })
            
            current_step += 1
            
        print(f"\r>> Progress: {current_step}/{total_steps} (Distance: {r}m 완료)", end="")

    print("\n>> 실험 완료! 데이터 분석 중...")
    return pd.DataFrame(results)

# --- 실행 및 시각화 ---
if __name__ == "__main__":
    df = run_automated_experiment()
    
    df.to_csv("./data/experiment_data.csv", index=False)
    print(">> 'experiment_data.csv' 저장 완료.")
    
    plt.figure(figsize=(12, 6))
    
    # 한글 폰트 설정 (Windows 기준)
    plt.rcParams['font.family'] = 'Malgun Gothic' 

    # 거리별 성공률 평균 계산
    summary = df.groupby("Distance")[["Success", "Confidence"]].mean()
    
    # 성공률 (파란선)
    sns.lineplot(data=summary, x="Distance", y="Success", marker="o", label="성공률 (Success Rate)")
    
    # 신뢰도 (주황선)
    sns.lineplot(data=summary, x="Distance", y="Confidence", marker="s", label="확신도 (Confidence)")
    
    plt.title("거리별 통신 강건성 실험 결과 (Inverse Square Law 적용)", fontsize=14, weight='bold')
    plt.ylabel("Score (0~1)")
    plt.xlabel("Distance (m)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.show()