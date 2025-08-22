import numpy as np

# 기존 .npy 파일 불러오기
try:
    original_array = np.load('image_feat.npy')
    original_shape = original_array.shape
    print(f"원본 배열의 크기 (shape): {original_shape}")

    # 동일한 크기의 랜덤 배열 생성
    # np.random.rand()는 0과 1 사이의 균일 분포 값을 생성합니다.
    random_array = np.random.rand(*original_shape)

    # 새로운 .npy 파일로 저장
    np.save('random_image_feat.npy', random_array)

    print(f"'random_image_feat.npy' 파일이 성공적으로 생성되었습니다.")
    print(f"새로운 배열의 크기 (shape): {random_array.shape}")

except FileNotFoundError:
    print("오류: 'image_feat.npy' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")