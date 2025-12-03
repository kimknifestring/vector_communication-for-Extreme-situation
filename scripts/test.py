# ai 딸깍

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import platform
from matplotlib import font_manager, rc

# 우리가 만든 모듈 불러오기
from transmitter import sender, channel, receiver, DATA_LABELS

# --- [1] 실험 설정 ---
MAX_DISTANCE = 7     # 최대 거리 (m)
STEP_SIZE = 0.5         # 거리 증가 단위 (m)
TRIALS_PER_STEP = 10    # 각 거리마다 반복할 횟수 (많을수록 그래프가 부드러워짐)

def set_korean_font():
    """OS에 맞게 한글 폰트를 자동으로 설정하는 함수"""
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif system_name == 'Darwin': # Mac
        plt.rcParams['font.family'] = 'AppleGothic'
    else: # Linux
        plt.rcParams['font.family'] = 'NanumGothic'
    
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

def run_experiment():
    print("=" * 60)
    print(f"🧪 [Final Experiment] 거리별 통신 강건성 정밀 테스트")
    print(f"   - 범위: 1.0m ~ {MAX_DISTANCE}m (간격 {STEP_SIZE}m)")
    print(f"   - 반복: 구간당 {TRIALS_PER_STEP}회")
    print("=" * 60)

    results = []
    distances = np.arange(1.0, MAX_DISTANCE + STEP_SIZE, STEP_SIZE)
    total_steps = len(distances) * TRIALS_PER_STEP
    current_step = 0

    for r in distances:
        for _ in range(TRIALS_PER_STEP):
            # 1. 타겟 선정 및 송신
            target_id = random.randint(0, 29)
            tx_vector = sender(target_id)
            
            # 2. 채널 통과 (일반 신호 / DAE 복원 신호 동시 획득)
            REPEAT_COUNT = 5 # 5번 반복
            rx_raw_sum = 0
            rx_dae_sum = 0

            for _ in range(REPEAT_COUNT):
                # 매번 새로운 노이즈가 섞임
                temp_raw, temp_dae = channel(tx_vector, distance_r=r)
                rx_raw_sum += temp_raw
                rx_dae_sum += temp_dae

            # 평균 계산 (신호는 남고 노이즈는 줄어듦)
            rx_raw = rx_raw_sum / REPEAT_COUNT
            rx_dae = rx_dae_sum / REPEAT_COUNT
            
            # 3. [Method A] Standard (No DAE)
            id_raw, _, conf_raw = receiver(rx_raw)
            success_raw = 1 if (target_id == id_raw) else 0
            
            results.append({
                "Distance": r,
                "Method": "Standard (No DAE)",
                "Success": success_raw,
                "Confidence": conf_raw
            })
            
            # 4. [Method B] DAE Filter (With DAE)
            id_dae, _, conf_dae = receiver(rx_dae)
            success_dae = 1 if (target_id == id_dae) else 0
            
            results.append({
                "Distance": r,
                "Method": "DAE Filter (With DAE)",
                "Success": success_dae,
                "Confidence": conf_dae
            })
            
            current_step += 1
            
        # 진행률 표시
        print(f"\r>> Progress: {current_step}/{total_steps} ({current_step/total_steps*100:.1f}%)", end="")

    print("\n>> 실험 완료! 데이터를 정리합니다.")
    return pd.DataFrame(results)

def plot_results(df):
    set_korean_font()
    
    # 캔버스 설정
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 색상 팔레트
    custom_palette = {"Standard (No DAE)": "#E74C3C", "DAE Filter (With DAE)": "#2E86C1"}

    sns.lineplot(
        data=df, x="Distance", y="Success", hue="Method",
        palette=custom_palette, style="Method", markers=True, dashes=False,
        linewidth=3,     # 선 두께를 좀 더 키워서 잘 보이게 함
        errorbar=None,
        ax=axes[0]
    )
    
    axes[0].set_title("거리별 통신 성공률 (Success Rate)", fontsize=16, weight='bold', pad=15)
    axes[0].set_ylabel("성공 확률", fontsize=12)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].axhline(0.9, color='gray', linestyle='--', alpha=0.5)
    axes[0].text(MAX_DISTANCE, 0.92, '90% Threshold', color='gray', ha='right')
    axes[0].legend(loc='lower left', frameon=True)
    axes[0].grid(True, alpha=0.4) # 격자를 조금 더 진하게
    sns.lineplot(
        data=df, x="Distance", y="Confidence", hue="Method",
        palette=custom_palette, style="Method", markers=True, dashes=False,
        linewidth=3,     # 선 두께 키움
        ax=axes[1]
    )
    
    axes[1].set_title("모델 확신도 변화 (Confidence Score)", fontsize=16, weight='bold', pad=15)
    axes[1].set_xlabel("물리적 거리 (m)", fontsize=12)
    axes[1].set_ylabel("코사인 유사도", fontsize=12)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(loc='upper right', frameon=True)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("final_result_clean.png", dpi=300)
    print(">> 깔끔한 그래프 저장 완료: final_result_clean.png")
    plt.show()

# --- 메인 실행 ---
if __name__ == "__main__":
    # 1. 실험 실행
    df_result = run_experiment()
    
    # 2. 데이터 저장
    df_result.to_csv("final_experiment_data.csv", index=False)
    print(">> 데이터 저장 완료: final_experiment_data.csv")
    
    # 3. 그래프 그리기
    plot_results(df_result)