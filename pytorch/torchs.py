import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ==================================================================
# 예제 1: 기본 텐서(Tensor) 조작
# ==================================================================
def example_1_tensors():
    """파이토치 텐서의 기본적인 생성 및 조작 방법을 보여줍니다."""
    print("--- 예제 1: 기본 텐서 조작 ---")
    
    # 데이터로부터 텐서 생성
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print("데이터로부터 생성:\n", x_data)

    # Numpy 배열로부터 텐서 생성
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print("Numpy 배열로부터 생성:\n", x_np)

    # 다른 텐서의 속성을 상속받아 텐서 생성 (ones_like, rand_like)
    x_ones = torch.ones_like(x_data)
    print("Ones 텐서:\n", x_ones)
    x_rand = torch.rand_like(x_data, dtype=torch.float)
    print("Random 텐서:\n", x_rand)
    
    # 특정 모양(shape)으로 텐서 생성
    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)
    print(f"{shape} 모양의 Random 텐서:\n", rand_tensor)
    
    # 텐서의 속성 (shape, dtype, device)
    tensor = torch.rand(3, 4)
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")
    
    print("-" * 30 + "\n")


# ==================================================================
# 예제 2: 선형 회귀 (Linear Regression)
# ==================================================================
def example_2_linear_regression():
    """간단한 선형 회귀 모델을 파이토치로 구현합니다."""
    print("--- 예제 2: 선형 회귀 ---")
    
    # 데이터 생성
    x_train = torch.FloatTensor([[1], [2], [3]])
    y_train = torch.FloatTensor([[2], [4], [6]])
    
    # 모델 정의: nn.Linear(input_dim, output_dim)
    # y = Wx + b 형태의 선형 계층
    model = nn.Linear(1, 1)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.MSELoss() # 평균 제곱 오차 (Mean Squared Error)
    optimizer = optim.SGD(model.parameters(), lr=0.01) # 확률적 경사 하강법
    
    # 훈련 루프
    epochs = 1000
    for epoch in range(epochs):
        # 순전파 (Forward pass)
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        
        # 역전파 및 최적화 (Backward and optimize)
        optimizer.zero_grad() # 그래디언트 초기화
        loss.backward()       # 그래디언트 계산
        optimizer.step()      # 가중치 업데이트
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    # 훈련된 모델로 예측
    with torch.no_grad(): # 그래디언트 계산 비활성화
        predicted = model(torch.FloatTensor([[4]])).item()
        print(f"\nx=4일 때의 예측값: {predicted:.4f}")
        
    print("-" * 30 + "\n")


# ==================================================================
# 예제 3: XOR 문제 해결
# ==================================================================
class XORModel(nn.Module):
    """XOR 문제를 해결하기 위한 간단한 2층 신경망 모델"""
    def __init__(self):
        super(XORModel, self).__init__()
        # 텐서플로우의 Dense 레이어와 유사
        self.layer1 = nn.Linear(2, 10) # 입력 2, 은닉층 뉴런 10
        self.layer2 = nn.Linear(10, 1) # 은닉층 뉴런 10, 출력 1
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 순전파 로직 정의
        out = self.layer1(x)
        out = self.sigmoid(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        return out

def example_3_xor():
    """XOR 문제를 파이토치 신경망으로 해결합니다."""
    print("--- 예제 3: XOR 문제 해결 ---")
    
    # 데이터 생성
    x_data = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_data = torch.FloatTensor([[0], [1], [1], [0]])
    
    # 모델, 손실 함수, 옵티마이저 초기화
    model = XORModel()
    criterion = nn.BCELoss() # 이진 교차 엔트로피 (Binary Cross Entropy)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # 훈련 루프
    epochs = 10000
    for epoch in range(epochs):
        outputs = model(x_data)
        loss = criterion(outputs, y_data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            # 정확도 계산
            predicted = (outputs > 0.5).float()
            correct = (predicted == y_data).sum().item()
            accuracy = correct / y_data.shape[0]
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}')
    
    # 최종 예측
    with torch.no_grad():
        predicted = (model(x_data) > 0.5).float()
        print("\n최종 예측 결과:")
        for i in range(len(x_data)):
            print(f"Input: {x_data[i].tolist()}, Predicted: {predicted[i].item()}")
            
    print("-" * 30 + "\n")


# ==================================================================
# 예제 4: MNIST 손글씨 숫자 분류
# ==================================================================
class MNISTModel(nn.Module):
    """MNIST 분류를 위한 간단한 신경망 모델"""
    def __init__(self):
        super(MNISTModel, self).__init__()
        # 28*28 이미지를 784 크기의 1차원 벡터로 다룸
        self.layer1 = nn.Linear(28*28, 256)
        self.layer2 = nn.Linear(256, 10) # 0~9, 10개 클래스
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 입력 x를 1차원 벡터로 펼침
        x = x.view(-1, 28*28)
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        # CrossEntropyLoss는 내부적으로 Softmax를 포함하므로, 여기서는 Softmax를 적용하지 않음
        return out

def example_4_mnist():
    """MNIST 데이터셋을 이용해 손글씨 숫자를 분류합니다."""
    print("--- 예제 4: MNIST 손글씨 숫자 분류 ---")

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.ToTensor(), # 이미지를 텐서로 변환
        transforms.Normalize((0.5,), (0.5,)) # -1 ~ 1 범위로 정규화
    ])

    # 데이터셋 다운로드 및 로드
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
    
    # DataLoader: 데이터를 배치 단위로 묶어줌
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # 모델, 손실 함수, 옵티마이저
    model = MNISTModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 훈련 루프
    epochs = 5
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 순전파
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # 모델 평가
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'\n테스트 데이터에서의 정확도: {100 * correct / total:.2f} %')
    
    print("-" * 30 + "\n")


# ==================================================================
# 메인 실행 블록
# ==================================================================
if __name__ == '__main__':
    # 아래 함수 호출의 주석을 해제하여 원하는 예제를 실행할 수 있습니다.
    
    # example_1_tensors()
    # example_2_linear_regression()
    # example_3_xor()
    example_4_mnist()
    
    print("모든 선택된 예제 실행 완료.")
