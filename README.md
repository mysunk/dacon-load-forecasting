공공 데이터 활용 전력수요 및 SMP 예측 AI 경진대회
=======================================

기상, 전력수급실적, SMP 데이터를 활용하여 일주일 후 부터 28일 간의 전력 수요량과 SMP를 예측하는 대회입니다.
대회 링크:
https://www.dacon.io/competitions/official/235606/overview/description/

Dataset
==================
이 저장소에 데이터셋은 제외되어 있습니다.  
데이터셋 출처: 
https://www.dacon.io/competitions/official/235606/data/

Structure
==================
```setup
.
└── main.py
└── evaluation.py
└── util.py
```
* main.py: feature extraction부터 modeling까지의 main문
* evaluation.py: validation 결과를 확인하고 ensemble하기 위한 파일
* util.py: custom 함수가 정의된 파일

Results
==================
* 평가지표: Weighted Root Mean Squared Scaled Error (WRMSSE)
* MSE 결과: 1.15
* private rank: 7/62