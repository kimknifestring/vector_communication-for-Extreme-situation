import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from train_denoiser import ResidualDenoiser

DENOISER_FILE = './data/denoiser_model.pth'
VECTOR_FILE = './data/robot_vectors_only_328d.pt'

CATEGORY_NAMES = {
    0: 'Move (이동)', 1: 'Rotate (회전)', 2: 'Control (제어)', 3: 'Manipulate (조작)',
    4: 'Sense (센싱)', 5: 'Status (상태)', 6: 'Signal (신호)', 7: 'Comm (통신)'
}

DATA_LABELS = [
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5,
    0, 0, 0, 6, 6, 7
]

def sender(action_id):
    if action_id < 0 or action_id >= 30:
        raise ValueError("!! [Error] 동작 ID는 0~29 사이의 정수 값입니다.")
    
    clean_vector = REFERENCE_VECTORS[action_id].clone()
    print(f"\n[Sender] ID {action_id}번 신호 생성 완료!")
    return clean_vector

def channel(signal_vector, distance_r):
    r = max(float(distance_r), 1.0)
    attenuation = 1.0 / (r ** 2)
    weak_signal = signal_vector * attenuation

    noise = torch.randn_like(signal_vector)
    distorted_signal = weak_signal + noise
    denoised_signal = denoiser(distorted_signal)

    print(f"[Channel] 거리 {r} 통과 중... (신호 강도 {attenuation * 100:.1f}% + 노이즈 혼합)")
    return distorted_signal, denoised_signal

def receiver(received_vector):
    similarities = F.cosine_similarity(received_vector.unsqueeze(0), REFERENCE_VECTORS)
    best_match_id = torch.argmax(similarities).item()
    confidence = similarities[best_match_id].item()

    predicted_category_idx = DATA_LABELS[best_match_id]
    category_name = CATEGORY_NAMES[predicted_category_idx]

    return best_match_id, category_name, confidence

try:
    REFERENCE_VECTORS = torch.load(VECTOR_FILE)
    print(f">> [System] 벡터 도감 로드 완료. 총 {len(REFERENCE_VECTORS)}개의 동작 대기 중.")

except FileNotFoundError:
    print(f"!! [Error] '{VECTOR_FILE}' 파일이 없습니다.  train.py를 먼저 실행해주세요.")
    exit()

try:
    denoiser = ResidualDenoiser()
    denoiser.load_state_dict(torch.load(DENOISER_FILE))
    denoiser.eval()
    print(">> [SYSTEM] DAE 모듈 불러오기 성공.")
except FileNotFoundError:
    print(f"!! [Error] '{DENOISER_FILE}' 파일이 없습니다.  train_denoiser.py를 먼저 실행해주세요.")
    exit()

if __name__ == "__main__":

    print("=" * 60)
    print(" [짱 멋지고 정밀한 벡터 통신기] 가동됨")
    print("   - 사용법: [동작ID] [거리] 를 입력하세요.")
    print("   - 예시: '0 5' -> 0번 명령을 5(가상 단위) 거리에서 전송")
    print("   - 종료: 'q' 입력")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n>> 명령 입력 (ID Distance)")

            if user_input.lower() == "q":
                print(">> 시뮬레이터를 종료합니다.")
            
            inputs = user_input.split()
            if len(inputs) != 2:
                print("!! [경고] 입력 형식을 준수해 주세요.")
                continue

            target_id = int(inputs[0])
            distance = float(inputs[1])

            tx_signal = sender(target_id)
            rx_signal, dn_signal = channel(tx_signal, distance)
            signal_list = (rx_signal, dn_signal)

            for signal in range(2):
                decoded_id, decoded_cat, confidence = receiver(signal_list[signal])

                is_success = (target_id == decoded_id)

                print("일반 수신" if signal == 0 else "DAE 수신")
                result_icon = "성공" if is_success else "실패"
                
                print("-" * 40)
                print(f"1. 보낸 명령: {target_id}번")
                print(f"2. 통신 환경: 거리 {distance} distance")
                print(f"3. 수신기 해석: {decoded_id}번 ({decoded_cat})")
                print(f"4. 확신 점수: {confidence:.4f} (0~1)")
                print(f"5. 최종 판정: {result_icon}")
                print("-" * 40)

        except ValueError:
            print("!! [에러] 숫자를 정확히 입력해주세요.")
        except Exception as e:
            print(f"!! [시스템 에러] {e}")