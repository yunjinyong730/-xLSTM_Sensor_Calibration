# TimeXer_Sensor_Calibration

```
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_2 (InputLayer)        [(1, 360, 5)]             0         
                                                                 
 normalizer_1 (Normalizer)   (1, 360, 5)               0         
                                                                 
 timexer (TimeXer)           (1, 1)                    6745      
                                                                 
 denormalizer (Denormalizer  (1, 1)                    0         
 )                                                               
...
_________________________________________________________________
934/934 [==============================] - 2s 1ms/step
Inference time: 1.866 seconds
Throughput: 16016.67 samples/second
```

<img width="1000" height="400" alt="Antwerp_pm10_w360" src="https://github.com/user-attachments/assets/b628bbd7-070d-4f65-8a87-da5c7bcf4b41" />   <br>
<b>val rmse : 8.589814186096191, test rmse : 13.827042579650879</b>
<img width="1000" height="400" alt="oslo_pm10_w360" src="https://github.com/user-attachments/assets/ca9bcb51-41d1-4eed-8ee9-b2369c3bc798" />   <br>
<b>val rmse : 9.852630615234375, test rmse : 15.736166954040527</b>
<img width="1000" height="400" alt="Zagreb_pm10_w360" src="https://github.com/user-attachments/assets/9f4fd5ad-93f4-481f-9e5b-7a158fcc4f55" />   <br>
<b>val rmse : 17.27320098876953, test rmse : 14.202249526977539</b>  <br>
<b>avg test rmse:  14.588486353556315 [13.827043, 15.736167, 14.20225]</b>



# TimeXer — 외생 변수(Exogenous Variables)를 통한 시계열 예측 강화 논문 정리

> **논문**: *TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables (NeurIPS 2024)*  
> **저자/소속**: Tsinghua Univ. BNRist  
> **핵심 아이디어 한 줄 요약**: **내생(Endogenous) 시계열은 패치 단위 토큰으로**, **외생(Exogenous) 시계열은 변수(variates) 단위 토큰으로** 표현하고, **글로벌 토큰**을 **다리**로 삼아 **패치-자기어텐션**과 **변수-교차어텐션**을 동시에 수행해 외부 요인을 견고하게 흡수한다.

---

## 목차
- [왜 외생 변수가 중요한가?](#왜-외생-변수가-중요한가)
- [문제 정의](#문제-정의)
- [모델 아키텍처](#모델-아키텍처)
- [학습/손실 및 멀티변수 예측으로의 일반화](#학습손실-및-멀티변수-예측으로의-일반화)
- [결과 요약](#결과-요약)
- [일반성/견고성/확장성](#일반성견고성확장성)
- [재현을 위한 기본 설정](#재현을-위한-기본-설정)
- [데이터셋 개요](#데이터셋-개요)
- [제한점과 팁](#제한점과-팁)
- [인용](#인용)

---

## 왜 외생 변수가 중요한가?
실세계 시계열은 **결측**, **비균일 샘플링**, **주기/길이 불일치**, **시간 지연 효과**가 흔하다. 기존 접근(내·외생을 동일 시점에 concat)으로는 **정렬/동기화**가 어렵고, 불필요한 상호작용과 복잡도가 커진다. TimeXer는 **임베딩 단계에서 역할을 분리**해 이러한 문제를 우회한다.

---

## 문제 정의
- **입력**: 내생 단변량 $x_{1:T}$ 와 다수의 외생 변수 집합 $z^{(1)}{1:T{\mathrm{ex}}}, \dots, z^{(C)}{1:T{\mathrm{ex}}}$
  (내·외생의 **look-back 길이 불일치** 허용, $T \neq T_{\mathrm{ex}}$)
- **목표**: 향후 $S$ 스텝의 내생 시계열 $\hat{x}{T+1:T+S} = F{\theta}!\big(x_{1:T}, z_{1:T_{\mathrm{ex}}}\big)$ 예측

---

## 모델 아키텍처
<img width="1041" height="454" alt="image" src="https://github.com/user-attachments/assets/63d05fe7-dc4d-4f53-84b1-53e47a551862" />

**핵심 설계**:  
1) **내생 임베딩(Endogenous)** — **비중첩 패치**로 나눈 뒤, **패치 토큰들**(temporal patch tokens) + **학습형 글로벌 토큰**(series-level global token) 구성. 글로벌 토큰이 **패치↔외생** 정보 통로 역할.  
2) **외생 임베딩(Exogenous)** — **변수(variates) 단위 시계열 전체**를 **하나의 토큰**으로 임베딩(**variate token**). 결측/미정렬/주기·길이 상이성에 **자연 적응**.  
3) **어텐션 흐름**  
   - **내생 자기어텐션(Self-Attn)**: [패치 토큰들 + 글로벌 토큰]에 대해 **패치-패치** 및 **패치-글로벌** 관계를 동시에 학습해 **시간 의존성**을 정확히 캡처.  
   - **외생→내생 교차어텐션(Cross-Attn)**: **내생 글로벌 토큰(질의)**이 **외생 변수 토큰들(키/값)**을 선택적으로 흡수 → **변수-수준 상관** 반영.  

> 직관: 외생은 “**무엇이 중요한 변수인가**”를 고르고(변수-수준), 내생은 “**언제 중요한가**”를 정밀히 본다(패치-수준). 두 축을 **글로벌 토큰**으로 엮어 **불필요한 전-변수간 상호작용 비용**을 줄이면서도 정보는 **선택적으로 유입**된다.

---

## 학습/손실 및 멀티변수 예측으로의 일반화
- **출력 생성**: 마지막 블록에서 얻은 패치 표현과 전역(글로벌) 표현을 하나로 합친 뒤, 이를 **선형 변환(완전연결층)**에 통과시켜 미래 값을 예측한다. 즉, 시간 구간별 정보(패치)와 시계열 전반의 요약 정보(전역)를 결합해 최종 예측을 만든다.
- **손실**: **L2(제곱 오차)**  
- **멀티변수 예측**: 각 변수를 “내생”으로 두고 **나머지 변수는 외생으로 병렬 처리**(채널 독립), **Self/Cross-Attn 층 공유**.

---

## 결과 요약
- **단기 전력가격(EPF, 입력 168→예측 24)** 5개 마켓 모두에서 **SOTA**(MSE/MAE). 예:  
  **PJM MSE 0.093** (iTransformer 0.097, Crossformer 0.101 등)  
  **NP MSE 0.236** (Crossformer 0.240, RLinear 0.335 등)  
  → 외생 변수의 **정확한 활용 + 시간 의존 학습**이 경쟁 모델을 일관되게 상회.  
- **장기 멀티변수(ETT/ECL/Weather/Traffic 등, 평균)** 대다수 데이터셋에서 **일관된 우수 성능**.  
- **왜 잘 되나?** 기존 모델은  
  - Crossformer: **모든 변수를 세밀 패치 수준**으로 엮어 **노이즈/복잡도 증가**  
  - iTransformer: **변수-수준만** 보고 **시간-세부**는 선형 투영에 의존  
  → TimeXer는 **패치(시간)×변수(외생) 이원 설계**로 장단점을 동시에 보완.

---

## 일반성/견고성/확장성
- **Look-back 불일치**(내생/외생 길이 다름)에도 **성능 이득 유지**. 외생 길이 확장보다 **내생 길이 확장이 특히 유익**.  
- **결측/랜덤 외생**에도 **내생의 시간 표현이 예측을 주도**해 **성능 강건**(외생이 완전히 무의미해도 급락하지 않음). 반대로 **내생이 무의미**해지면 급격히 악화.  
- **효율성**: 외생 간 상호작용을 층마다 풀어놓지 않고 **글로벌 토큰 기반 교차어텐션**으로 처리 → **메모리 우위/학습속도 유리**.

---

## 재현을 위한 기본 설정
- **프레임워크/하드웨어**: PyTorch, 단일 RTX 4090 24GB  
- **최적화**: Adam, **lr=1e-4**, L2 Loss, **Early Stopping**, **10 epoch** 고정 학습  
- **모델 크기**: Block $L \in \{1,2,3\}$, $d_{\text{model}}\in\{128,256,512\}$
- **패치 길이**: **장기 16**, **단기 24(비중첩)** — 작은 패치는 의미 정보 희석 가능(성능 저하)

### Quick Pseudo-code (개념 흐름)
```python
# x: endogenous (T,), z_list: [z^(1)_(T_ex), ..., z^(C)_(T_ex)]
patch_tokens = PatchEmbed(split_nonoverlap(x, P))       # (N, D)
g_token     = LearnableGlobalToken()                    # (1, D)
v_tokens    = [VariateEmbed(z) for z in z_list]         # (C, D)

# L layers
for _ in range(L):
    # Self-Attn over [patch_tokens || g_token]
    patch_tokens, g_token = SelfAttentionConcat(patch_tokens, g_token)
    # Cross-Attn: g_token (Q)  <-- v_tokens (K,V)
    g_token = CrossAttention(g_token, v_tokens)

y_hat = LinearProjection(concat(patch_tokens, g_token))  # forecast
loss  = mse(y_hat, y_true)
