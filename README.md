# Vision_Transformer-implementation-Pytorch
ViT를 구현하기 위해 정리하는 Repository

## ViT 장점
- transformer 구조를 거의 그대로 사용하기 때문에 확장성이 좋음
- Large 스케일 학습에서 매우 우수한 성능을 보임
- transfer learning 시 CNN 보다 훈련에 더 적은 계산 리소스 사용.

## ViT 단점

- Inductive bias의 부족으로 인해 CNN보다 데이터가 많이 요구

### Inductive bias란? 

학습자가 처음보는 입력에 대한 출력을 예측하기 위해 사용하는 일련의 가정( assumbtion )

CNN은 입력에 대한 출력을 예측하기 위해 몇 가지 가정을 하는데, 대표적으로 Translation equivariance랑 locality 가정을 합니다.

locality라는건 우리가 합성곱을 연산할 때 이 이미지 전체에서 일부만 이렇게 보잖아요. 즉 이말은 우리가 특정 영역만을 보고 그 안에서 특징을 추출할 수 있다는 것을 가정한거에요. 즉 입력에 대해서 출력을 예측하기 위해 하나의 가정을 만들어 연산을 한 게 합성곱 연산.

Translation equivariance라는 건 입력 위치가 변하면 출력도 동일한 위치로 변한 채 나온다.
CNN은 위치가 변하면 출력값의 위치도 같이 변하면서 해당 값을 유지시켜준다. 즉, 순서는 다르지만 추출된 피처가 보존되고 있다는 것. Translation equivariance라는 성질을 가지고 있기 때문에 MLP보다 성능이 좋은 것. MLP같은 경우 이미지를 일렬로 펴서 리니어를 거치죠? 리니어 레이어를 거치고 다시 리니어 레이어 형태의 출력값을 뽑아내게 되는데 이렇게 되면 Translation equivariance가 보장되지않아 가중치들이 달라지기 때문에 결과값이 완전히 다르게 나올수도 있는 것.

이미지 처리에서는 CNN이 MLP보다 좋은 이유가 저것.

어쨌든 CNN은 Inductive bias를 가지고 정해진 성질에 의해 학습이 진행됩니다. 그래서 이제 CNN은 강력한 가정들을 가지고 있고 하지만 트랜스포머 같은 경우에는 attention만 사용합니다.

attention 개념 자체가 전체를 보고 어디가 어떤지를 말하는 모델이기 때문에 어디를 어떻게 봐라 같은 가정들이 부족하니까 그 패턴을 전체적인 부분에서 찾기 때문에 당연히 CNN보다 더 많은 데이터가 필요로 한다는 것입니다.

그렇기 때문에 불충분한 양의 데이터로 훈련을 하면 잘 일반화가 되지 않습니다.




예를들어 중간 사이즈인 이미지넷을 강한 정규화 없이 학습에 사용할 경우 유사한 크기의 ResNet보다 성능이 낮음. 그러나 라지 스케일로 학습을 했더니 강한 Inductive bias를 가진 CNN을 능가한다는 점을 발견.

충분한 스케일에서 사전 학습을 하고 적은 데이터로 전이하는 방식을 채택했더니 전이 학습 속도도 CNN보다 빠르고 정확도도 더 높았다는 겁니다.


실제 논문에서는 라지스케일인 이미지넷 21K나 JFT 300M으로 사전학습을 했구요. 그 다음 CIFAR같은 다양한 데이터로 전이학습을 했는데 최신 모델의 정확도를 능가하거나 근접한 정확도를 달성했습니다.



# ViT 학습

- large 데이터셋으로 사전학습 후 더 작은 데이터셋에 대해 fine-tune 하는 방식

  ( 이미지 resize 및 MLP 헤드 부분을 클래스 수에 맞게 교체 )

- 학습을 위해 large 데이터셋인 ImageNet, ImageNet-21k, JFT사용
- 전처리는 Resize, RandomCrop, RandomHorizontalFilp 사용
- 광범위하게 Dropout 적용 ( qkv-prediction 부분 제외 )
- 아래 데이터셋에 대해 전이학습 진행 

  ( ImageNet, CIFAR10/100, 9-task VTAB 등 )
