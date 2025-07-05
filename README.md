# Stock Analysis Project

주식 데이터 분석 및 예측 프로젝트입니다.

## 📁 프로젝트 구조

```
stock/
├── etc/
│   ├── stock.py          # 주식 분석 메인 로직
│   ├── predict.py        # 예측 모델
│   ├── report.py         # 리포트 생성
│   ├── mig.sh           # 마이그레이션 스크립트
│   ├── fred_data.csv    # FRED 경제 데이터
│   ├── fred_data_yf.csv # FRED + Yahoo Finance 데이터
│   ├── fred_data_yf_stock.csv # 주식 통합 데이터
│   └── total.csv        # 전체 통합 데이터
└── README.md
```

## 🚀 주요 기능

- **데이터 수집**: FRED 및 Yahoo Finance에서 주식 데이터 수집
- **데이터 분석**: 주식 가격 및 경제 지표 분석
- **예측 모델**: 머신러닝을 활용한 주식 가격 예측
- **리포트 생성**: 분석 결과 자동 리포트 생성

## 📊 데이터 소스

- **FRED (Federal Reserve Economic Data)**: 경제 지표 데이터
- **Yahoo Finance**: 실시간 주식 데이터

## 🛠️ 사용 기술

- Python
- Git
- CSV 데이터 처리
- 머신러닝 (예측 모델)

## 📝 사용법

1. `etc/` 폴더의 Python 스크립트들을 실행
2. 데이터 분석 및 예측 수행
3. 리포트 확인

---

*Last updated: 2024*
