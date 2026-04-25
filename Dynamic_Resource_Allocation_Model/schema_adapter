# -*- coding: utf-8 -*-
"""
schema_adapter.py — 입력 문제 완전 해결 (단점 1, 2, 3)
────────────────────────────────────────────────────────────
SmartCSVLoader:
  - 컬럼명이 달라도 자동 감지·매핑 (target_rps, rps, requests, throughput 등)
  - 요일/이벤트 피처 없는 CSV도 자동 생성
  - 어떤 CSV든 표준 6피처로 변환 보장
  - 입력 데이터 요약 리포트 출력
"""

import os
import warnings
import numpy as np
import pandas as pd

# ── 표준 피처 스키마 ──────────────────────────────────────────────────────
STANDARD_FEATURES = ['target_rps', 'diff', 'diff2', 'day_of_week', 'is_event', 'is_weekend']

# ── RPS 컬럼 후보 이름들 (우선순위 순) ────────────────────────────────────
RPS_COLUMN_ALIASES = [
    'target_rps', 'rps', 'requests_per_second', 'throughput',
    'request_rate', 'qps', 'tps', 'traffic', 'load',
    'value', 'count', 'requests', 'hits', 'rate',
]

# ── 요일 컬럼 후보 이름들 ─────────────────────────────────────────────────
DOW_COLUMN_ALIASES = [
    'day_of_week', 'dow', 'weekday', 'day', 'day_num',
]

# ── 이벤트 컬럼 후보 이름들 ──────────────────────────────────────────────
EVENT_COLUMN_ALIASES = [
    'is_event', 'event', 'is_sale', 'sale', 'promotion', 'campaign',
    'is_promotion', 'special',
]

# ── 주말 컬럼 후보 이름들 ─────────────────────────────────────────────────
WEEKEND_COLUMN_ALIASES = [
    'is_weekend', 'weekend', 'is_sat_sun',
]

# ── 이벤트로 간주할 키워드 (파일명 / scenario 컬럼) ──────────────────────
EVENT_KEYWORDS = ['sale', 'event', 'promo', 'campaign', 'special', 'flash', 'black_friday']


class SchemaAdaptError(Exception):
    """CSV 스키마 자동 변환 실패 시 발생"""
    pass


class SmartCSVLoader:
    """
    어떤 CSV 파일이든 표준 6피처 DataFrame으로 자동 변환.

    사용 예:
        loader = SmartCSVLoader('my_traffic.csv')
        df     = loader.load()          # 표준 스키마 DataFrame 반환
        loader.print_report()           # 변환 요약 리포트 출력
    """

    def __init__(self, csv_path: str, verbose: bool = True):
        self.csv_path = csv_path
        self.verbose  = verbose
        self.report   = {}          # 변환 내역 저장
        self._df_raw  = None

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """CSV → 표준 스키마 DataFrame 반환 (항상 STANDARD_FEATURES 컬럼 보장)"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"파일 없음: {self.csv_path}")

        self._df_raw = pd.read_csv(self.csv_path)
        df = self._df_raw.copy()

        self.report['file']       = os.path.basename(self.csv_path)
        self.report['raw_shape']  = df.shape
        self.report['raw_cols']   = list(df.columns)
        self.report['mappings']   = {}
        self.report['generated']  = []
        self.report['warnings']   = []

        df = self._normalize_columns(df)
        df = self._build_rps(df)
        df = self._build_day_of_week(df)
        df = self._build_is_event(df)
        df = self._build_is_weekend(df)
        df = self._build_diff_features(df)
        df = self._validate_and_clean(df)

        # 최종 표준 컬럼만 반환
        result = df[STANDARD_FEATURES].copy()

        self.report['final_shape'] = result.shape
        self.report['rps_stats']   = {
            'min':  round(float(result['target_rps'].min()), 1),
            'max':  round(float(result['target_rps'].max()), 1),
            'mean': round(float(result['target_rps'].mean()), 1),
            'std':  round(float(result['target_rps'].std()), 1),
        }

        if self.verbose:
            self.print_report()

        return result

    def print_report(self):
        """변환 요약 리포트를 터미널에 출력"""
        r = self.report
        print(f"\n{'─'*55}")
        print(f"📋 SmartCSVLoader 변환 리포트: {r.get('file','?')}")
        print(f"{'─'*55}")
        print(f"  원본 shape : {r.get('raw_shape','?')}")
        print(f"  원본 컬럼  : {r.get('raw_cols','?')}")
        print(f"  최종 shape : {r.get('final_shape','?')}")
        print(f"  RPS 통계   : min={r['rps_stats']['min']}  max={r['rps_stats']['max']}"
              f"  mean={r['rps_stats']['mean']}  std={r['rps_stats']['std']}")

        if r.get('mappings'):
            print(f"\n  [컬럼 매핑]")
            for k, v in r['mappings'].items():
                print(f"    {v}  →  {k}")

        if r.get('generated'):
            print(f"\n  [자동 생성 피처]")
            for g in r['generated']:
                print(f"    ✦ {g}")

        if r.get('warnings'):
            print(f"\n  [⚠️  경고]")
            for w in r['warnings']:
                print(f"    {w}")

        print(f"{'─'*55}\n")

    # ─────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """컬럼명을 소문자+언더스코어로 정규화"""
        df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_')
                      for c in df.columns]
        return df

    def _find_column(self, df: pd.DataFrame, aliases: list) -> str | None:
        """aliases 목록 중 df에 존재하는 첫 번째 컬럼명 반환"""
        for alias in aliases:
            if alias in df.columns:
                return alias
        return None

    def _build_rps(self, df: pd.DataFrame) -> pd.DataFrame:
        """target_rps 컬럼 보장"""
        found = self._find_column(df, RPS_COLUMN_ALIASES)

        if found is None:
            # 숫자 컬럼 중 값 범위가 가장 RPS스러운 것 자동 선택
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # time_sec 류 제외
            exclude_keywords = ['time', 'sec', 'min', 'hour', 'idx', 'index', 'id']
            candidates = [c for c in num_cols
                          if not any(kw in c for kw in exclude_keywords)]
            if not candidates:
                raise SchemaAdaptError(
                    f"RPS에 해당하는 숫자 컬럼을 찾을 수 없습니다.\n"
                    f"컬럼 목록: {list(df.columns)}\n"
                    f"힌트: 컬럼명을 {RPS_COLUMN_ALIASES[:5]} 중 하나로 바꿔주세요."
                )
            # 평균값이 가장 큰 컬럼을 RPS로 추정
            found = max(candidates, key=lambda c: df[c].mean())
            self.report['warnings'].append(
                f"RPS 컬럼을 찾지 못해 '{found}'을 자동 선택했습니다. "
                f"맞지 않으면 컬럼명을 'target_rps'로 변경하세요."
            )

        if found != 'target_rps':
            df = df.rename(columns={found: 'target_rps'})
            self.report['mappings']['target_rps'] = found

        # 숫자 변환 + 음수 제거
        df['target_rps'] = pd.to_numeric(df['target_rps'], errors='coerce').fillna(0)
        df['target_rps'] = df['target_rps'].clip(lower=0)

        return df

    def _build_day_of_week(self, df: pd.DataFrame) -> pd.DataFrame:
        """day_of_week 컬럼 보장 (0=월 … 6=일)"""
        found = self._find_column(df, DOW_COLUMN_ALIASES)

        if found is not None:
            if found != 'day_of_week':
                df = df.rename(columns={found: 'day_of_week'})
                self.report['mappings']['day_of_week'] = found
            df['day_of_week'] = (pd.to_numeric(df['day_of_week'], errors='coerce')
                                   .fillna(0).astype(int) % 7)
        else:
            # time_sec 컬럼이 있으면 누적 초 → 요일 유추 시도
            if 'time_sec' in df.columns:
                seconds_per_day = 2001          # 데이터 관찰 기반 (2001개 = 1일)
                df['day_of_week'] = (df.index // seconds_per_day % 7).astype(int)
                self.report['generated'].append(
                    "day_of_week (time_sec 인덱스 기반 자동 추정 — 정밀도 낮음)"
                )
                self.report['warnings'].append(
                    "요일 정보를 자동 추정했습니다. 정확한 요일 컬럼을 제공하면 성능이 향상됩니다."
                )
            else:
                df['day_of_week'] = 0
                self.report['generated'].append("day_of_week = 0 (요일 정보 없음, 기본값)")

        return df

    def _build_is_event(self, df: pd.DataFrame) -> pd.DataFrame:
        """is_event 컬럼 보장"""
        found = self._find_column(df, EVENT_COLUMN_ALIASES)
        fname = os.path.basename(self.csv_path).lower()

        if found is not None:
            if found != 'is_event':
                df = df.rename(columns={found: 'is_event'})
                self.report['mappings']['is_event'] = found
            df['is_event'] = pd.to_numeric(df['is_event'], errors='coerce').fillna(0).astype(int)
        else:
            # 파일명 키워드로 판단
            is_evt_by_name = int(any(kw in fname for kw in EVENT_KEYWORDS))
            # scenario 컬럼이 있으면 이벤트 시나리오로 간주
            is_evt_by_scenario = int('scenario' in df.columns or 'phase' in df.columns)
            is_evt = max(is_evt_by_name, is_evt_by_scenario)
            df['is_event'] = is_evt
            basis = 'phase/scenario 컬럼 감지' if is_evt_by_scenario else \
                    '파일명 키워드 감지' if is_evt_by_name else '기본값 0'
            self.report['generated'].append(f"is_event = {is_evt}  ({basis})")

        return df

    def _build_is_weekend(self, df: pd.DataFrame) -> pd.DataFrame:
        """is_weekend 컬럼 보장"""
        found = self._find_column(df, WEEKEND_COLUMN_ALIASES)

        if found is not None:
            if found != 'is_weekend':
                df = df.rename(columns={found: 'is_weekend'})
                self.report['mappings']['is_weekend'] = found
            df['is_weekend'] = pd.to_numeric(df['is_weekend'], errors='coerce').fillna(0).astype(int)
        else:
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            self.report['generated'].append("is_weekend (day_of_week 5,6 기반 자동 생성)")

        return df

    def _build_diff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """변화량(diff, diff2) 피처 생성"""
        df['diff']  = df['target_rps'].diff().fillna(0)
        df['diff2'] = df['diff'].diff().fillna(0)
        return df

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측값 처리 + 이상치 소프트 클리핑 + 최소 행 수 검증"""
        # 결측값
        df = df.ffill().bfill().fillna(0)

        # 이상치 소프트 클리핑 (99.5 퍼센타일)
        cap = df['target_rps'].quantile(0.995)
        clipped = (df['target_rps'] > cap).sum()
        if clipped > 0:
            df['target_rps'] = df['target_rps'].clip(upper=cap)
            self.report['warnings'].append(
                f"target_rps 상위 {clipped}개 샘플을 {cap:.0f} RPS로 클리핑했습니다."
            )

        if len(df) < 120:
            raise SchemaAdaptError(
                f"데이터가 너무 적습니다: {len(df)}행 (최소 120행 필요)"
            )

        return df
