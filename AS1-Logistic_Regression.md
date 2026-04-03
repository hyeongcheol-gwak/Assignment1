# Assignment 1 Report 

## 1. Explain each implementation (1-1 to 1-3) and compare the differences in the code produced by the two LLMs.

### [Problem 1-1]
- **Used LLM_1** :  Gemini

- **Code** :  
```python
def learn_mul(X, y):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    return lr

def inference_mul(x, lr_model):
    y_pred = lr_model.predict(x)
    return y_pred
```
- **Code Description** :  
Gemini의 구현은 sklearn의 LogisticRegression을 기본 설정(solver='lbfgs', multi_class='auto')으로 사용하며, `max_iter=1000`으로 최대 반복 횟수를 설정하여 수렴을 보장한다. 추론 단계에서는 `predict()` 메서드를 직접 호출하여 가장 높은 확률의 클래스 레이블을 바로 반환하는 간결한 방식을 사용한다.



---

- **Used LLM_2** :  Claude

- **Code** :  
```python
def learn_mul(X, y):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(solver='saga', max_iter=2000, multi_class='multinomial', C=1.0)
    lr.fit(X, y)
    return lr

def inference_mul(x, lr_model):
    import numpy as np
    proba = lr_model.predict_proba(x)
    y_pred = np.argmax(proba, axis=1)
    return y_pred
```
- **Code Description** :  
Claude의 구현은 Gemini와 달리 `solver='saga'`와 `multi_class='multinomial'`을 명시적으로 지정하여 Softmax 기반의 다중 클래스 분류를 수행한다. SAGA solver는 대규모 데이터셋에 적합한 확률적 경사법 기반 최적화 알고리즘이다. 또한 정규화 강도 `C=1.0`을 명시적으로 설정하고, `max_iter=2000`으로 더 많은 반복을 허용한다. 추론 단계에서도 `predict()` 대신 `predict_proba()`로 클래스별 확률을 얻은 뒤 `np.argmax`로 가장 높은 확률의 클래스를 선택하는 방식으로, Gemini보다 더 명시적인 접근법을 취한다.


---
### [Problem 1-2]
- **Used LLM_1** :  Gemini

- **Code** :  
```python
def learn_ovr(X, y):
    lrs = []
    for i in range(num_classes):
        from sklearn.linear_model import LogisticRegression
        y_binary = (y == i).astype(int)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y_binary)
        lrs.append(lr)
    return lrs

def inference_ovr(X, lrs):
    import numpy as np
    X_2d = np.atleast_2d(X)
    probas = []
    for lr in lrs:
        probas.append(lr.predict_proba(X_2d)[:, 1])
    y_pred = np.argmax(probas, axis=0)
    if np.ndim(X) == 1:
        return y_pred[0]
    return y_pred
```
- **Code Description** :  
Gemini의 OvR 구현은 각 클래스에 대해 `(y == i).astype(int)` 방식으로 이진 레이블을 생성하고, 기본 설정의 LogisticRegression을 학습시킨다. 추론 시에는 각 분류기의 `predict_proba()`에서 양성(positive) 클래스에 대한 확률([:, 1])을 수집한 뒤, `np.argmax`로 가장 높은 확률을 가진 클래스를 최종 예측으로 선택한다. 1차원 입력에 대한 처리도 포함되어 있다.



---

- **Used LLM_2** :  Claude

- **Code** :  
```python
def learn_ovr(X, y):
    lrs = []
    for i in range(num_classes):
        from sklearn.linear_model import LogisticRegression
        y_binary = np.where(y == i, 1, 0)
        lr = LogisticRegression(solver='lbfgs', max_iter=500, C=1.0)
        lr.fit(X, y_binary)
        lrs.append(lr)
    return lrs

def inference_ovr(X, lrs):
    import numpy as np
    X_2d = np.atleast_2d(X)
    scores = np.zeros((X_2d.shape[0], len(lrs)))
    for idx, lr in enumerate(lrs):
        scores[:, idx] = lr.decision_function(X_2d)
    y_pred = np.argmax(scores, axis=1)
    if np.ndim(X) == 1:
        return y_pred[0]
    return y_pred
```
- **Code Description** :  
Claude의 OvR 구현은 두 가지 핵심적인 차이점이 있다. 첫째, 이진 레이블 생성 시 `np.where(y == i, 1, 0)`을 사용하여 Gemini의 `.astype(int)` 방식과 기능적으로 동일하지만 구문적으로 다른 접근을 취한다. 둘째, solver를 `'lbfgs'`로, 정규화 강도 `C=1.0`을 명시적으로 지정한다. 추론 단계에서 가장 큰 차이가 나타나는데, Gemini가 `predict_proba()`로 확률 기반 비교를 하는 반면, Claude는 `decision_function()`을 사용하여 결정 경계까지의 부호 있는 거리(signed distance)를 기반으로 클래스를 선택한다. 또한 scores를 미리 할당된 numpy 배열에 저장하여 메모리 효율적인 접근을 취한다.


---
### [Problem 1-3]
- **Used LLM_1** :  Gemini

- **Code** :  
```python
def learn_ovo(X, y):
    lrs = {}
    class_pairs = list(combinations(range(num_classes), 2))
    for i, j in class_pairs:
        from sklearn.linear_model import LogisticRegression
        mask = (y == i) | (y == j)
        X_sub, y_sub = X[mask], y[mask]
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_sub, y_sub)
        lrs[(i, j)] = lr
    return lrs

def inference_ovo(X, lrs):
    import numpy as np
    X_2d = np.atleast_2d(X)
    num_classes = max([max(pair) for pair in lrs.keys()]) + 1
    votes = np.zeros((X_2d.shape[0], num_classes))
    for (_, _), lr in lrs.items():
        preds = lr.predict(X_2d).astype(int)
        np.add.at(votes, (np.arange(X_2d.shape[0]), preds), 1)
    y_pred = np.argmax(votes, axis=1)
    if np.ndim(X) == 1:
        return y_pred[0]
    return y_pred
```
- **Code Description** :  
Gemini의 OvO 구현은 boolean masking(`(y == i) | (y == j)`)으로 해당 클래스 쌍의 데이터를 필터링하여 학습한다. 추론 단계에서는 **다수결 투표(Majority Voting)** 방식을 사용한다. 각 분류기가 `predict()`로 예측한 클래스 레이블에 대해 `np.add.at()`를 사용하여 투표 행렬에 1표씩 누적하고, 최종적으로 가장 많은 표를 받은 클래스를 선택한다. 이 방식은 각 분류기의 예측에 동일한 가중치를 부여하는 단순하고 직관적인 방법이다.



---

- **Used LLM_2** :  Claude

- **Code** :  
```python
def learn_ovo(X, y):
    lrs = {}
    class_pairs = list(combinations(range(num_classes), 2))
    for i, j in class_pairs:
        from sklearn.linear_model import LogisticRegression
        idx = np.where((y == i) | (y == j))[0]
        X_pair = X[idx]
        y_pair = y[idx]
        lr = LogisticRegression(solver='lbfgs', max_iter=500, C=1.0)
        lr.fit(X_pair, y_pair)
        lrs[(i, j)] = lr
    return lrs

def inference_ovo(X, lrs):
    import numpy as np
    X_2d = np.atleast_2d(X)
    n_samples = X_2d.shape[0]
    num_classes = max(max(k) for k in lrs.keys()) + 1
    votes = np.zeros((n_samples, num_classes))
    for (ci, cj), lr in lrs.items():
        proba = lr.predict_proba(X_2d)
        cls_list = lr.classes_.tolist()
        idx_i = cls_list.index(ci)
        idx_j = cls_list.index(cj)
        votes[:, ci] += proba[:, idx_i]
        votes[:, cj] += proba[:, idx_j]
    y_pred = np.argmax(votes, axis=1)
    if np.ndim(X) == 1:
        return y_pred[0]
    return y_pred
```
- **Code Description** :  
Claude의 OvO 구현은 학습 단계에서 `np.where()`로 인덱스를 명시적으로 추출하여 데이터를 필터링하며, solver와 정규화 파라미터를 명시적으로 지정한다. 추론 단계에서 Gemini와의 핵심적인 차이가 드러나는데, Gemini의 단순 다수결 투표와 달리 Claude는 **신뢰도 기반 가중 투표(Confidence-weighted Voting)** 방식을 사용한다. 각 분류기의 `predict_proba()`로 얻은 확률값을 해당 클래스에 누적하여, 예측 확신도가 높은 분류기의 의견이 더 큰 영향력을 갖도록 한다. 또한 `lr.classes_.tolist()`를 통해 모델이 학습한 클래스 순서를 확인하여 올바른 확률을 올바른 클래스에 매핑하는 안전한 처리가 추가되어 있다.


---
## 2.Compare and analyze OvR (1-2) and OvO (1-3) in terms of accuracy, time cost, and efficiency.

### 정확도 (Accuracy)

| 전략 | 모델 수 | 예측 방식 |
|------|---------|-----------|
| OvR | N개 (= 클래스 수 = 10) | 각 분류기의 양성 확률 중 최대값 선택 |
| OvO | N(N-1)/2개 (= 45) | 다수결 투표 또는 확률 기반 투표 |

- **OvO**는 각 분류기가 두 클래스 간의 경계만 학습하므로, 클래스 간 세밀한 차이를 더 잘 포착할 수 있어 일반적으로 약간 더 높은 정확도를 보인다.
- **OvR**은 한 클래스 vs 나머지 전체를 구분하므로 클래스 불균형 문제가 발생할 수 있다 (10개 클래스인 경우 양성:음성 = 1:9). 이로 인해 분류 경계가 다수 클래스 쪽으로 편향될 수 있다.

### 학습 시간 (Time Cost)

- **OvR**: 10개의 분류기를 학습하되, 각 분류기는 전체 학습 데이터(60,000개)를 사용하므로 총 학습 데이터 크기는 10 × 60,000 = **600,000 샘플**이다.
- **OvO**: 45개의 분류기를 학습하지만, 각 분류기는 2개 클래스의 데이터만 사용하므로, 클래스가 균등 분포인 경우 각 분류기는 약 12,000개 샘플을 사용한다. 총 학습 데이터 크기는 45 × 12,000 = **540,000 샘플**이다.
- 학습 시간은 데이터 크기에 비례하므로, Fashion-MNIST처럼 클래스가 균등 분포인 경우 두 방법의 총 학습 시간은 유사하다. 다만 OvO는 분류기 수가 4.5배 많아 반복 오버헤드가 추가될 수 있다.

### 효율성 (Efficiency)

| 비교 항목 | OvR | OvO |
|-----------|-----|-----|
| 분류기 수 | O(N) | O(N²) |
| 분류기당 학습 데이터 | 전체 데이터 | 2개 클래스 데이터 |
| 메모리 사용량 | 적음 | 많음 (45개 모델 저장) |
| 추론 시간 | 빠름 (10개 모델만 호출) | 느림 (45개 모델 호출) |
| 확장성 | 클래스 수 증가 시 유리 | 클래스 수 증가 시 불리 (N² 스케일링) |

**결론**: OvR은 분류기 수가 클래스 수에 선형적이므로 클래스 수가 많을 때 효율적이며, OvO는 클래스 수가 적을 때 각 분류기의 학습이 가벼워 유리하다. Fashion-MNIST(10 클래스)에서는 OvO의 O(N²) 스케일링이 아직 부담이 크지 않아 두 방법 모두 적용 가능하지만, 클래스 수가 크게 증가하면 OvR이 더 실용적인 선택이다.


---
## 3. Briefly describe the strategies or specific functions implemented to maximize performance.
1. **Adam Optimizer**: 기본적인 SGD 대신 Adam(Adaptive Moment Estimation) 최적화 알고리즘을 사용하였다. Adam은 1차 모멘트(평균)와 2차 모멘트(분산)를 추적하며 학습률을 파라미터별로 적응적으로 조절한다. SGD에 비해 학습률에 덜 민감하고 수렴 속도가 빠르다는 장점이 있다.
   - β1=0.9, β2=0.999, ε=1e-8의 표준 하이퍼파라미터를 사용
   - Bias correction을 적용하여 초기 학습 단계에서의 편향을 보정

2. **L2 Regularization (Ridge)**: `lambda_param=1.0`의 L2 정규화를 gradient에 직접 반영하여 과적합을 방지하였다.

3. **Polynomial Features (Degree 2)**: 2차 다항 특성을 수동으로 생성하여 입력 특성 간의 교호작용(interaction)을 모델에 반영하였다. 이를 통해 선형 모델의 비선형 결정 경계 표현 능력을 향상시켰다.

4. **Sigmoid Clipping**: `np.clip(z, -250, 250)`을 적용하여 수치적 안정성을 확보하였다.

5. **Zero Initialization**: 가중치를 0으로 초기화하여 학습을 시작하였다.
---