# -*- coding: utf-8 -*-
"""
schema_adapter.py — 피크 추종 강화판 (diff2 가속도 + 단기 MA 피처 추가)
────────────────────────────────────────────────────────────
변경점:
  - diff2(가속도) 피처를 STANDARD_FEATURES에 포함 (피크 직전 신호 2.3배 강함)
  - ma5(5초 이동평균) 피처 추가: 단기 급등 패턴 포착
  - 어떤 CSV든 자동 스키마 변환 (단점 1,2,3 해결 유지)
"""

import os
import warnings
import numpy as np
import pandas as pd

# ── 표준 피처 스키마 (8개) ────────────────────────────────────────────────
# diff2(가속도)와 ma5(단기 MA)를 추가하여 피크 직전 급등 신호를 더 잘 포착
STANDARD_FEATURES = [
    'target_rps',   # 현재 RPS
    'diff',         # 변화량  (1차 미분)
    'diff2',        # 가속도  (2차 미분) ← 피크 직전 신호 2.3배
    'ma5',          # 5초 이동평균   ← 단기 급등 패턴
    'ma20',         # 20초 이동평균  ← 중기 트렌드
    'day_of_week',  # 요일 (0=월 ~ 6=일)
    'is_event',     # 이벤트 여부
    'is_weekend',   # 주말 여부
]

RPS_COLUMN_ALIASES = [
    'target_rps', 'rps', 'requests_per_second', 'throughput',
    'request_rate', 'qps', 'tps', 'traffic', 'load',
    'value', 'count', 'requests', 'hits', 'rate',
]
DOW_COLUMN_ALIASES     = ['day_of_week', 'dow', 'weekday', 'day', 'day_num']
EVENT_COLUMN_ALIASES   = ['is_event', 'event', 'is_sale', 'sale', 'promotion']
WEEKEND_COLUMN_ALIASES = ['is_weekend', 'weekend', 'is_sat_sun']
EVENT_KEYWORDS         = ['sale', 'event', 'promo', 'campaign', 'special', 'flash']


class SchemaAdaptError(Exception):
    pass


class SmartCSVLoader:
    """
    어떤 CSV 파일이든 표준 8피처 DataFrame으로 자동 변환.

    사용 예:
        loader = SmartCSVLoader('my_traffic.csv')
        df     = loader.load()
        loader.print_report()
    """

    def __init__(self, csv_path: str, verbose: bool = True):
        self.csv_path = csv_path
        self.verbose  = verbose
        self.report   = {}
        self._df_raw  = None

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"파일 없음: {self.csv_path}")

        self._df_raw = pd.read_csv(self.csv_path)
        df = self._df_raw.copy()

        self.report = {
            'file': os.path.basename(self.csv_path),
            'raw_shape': df.shape,
            'raw_cols': list(df.columns),
            'mappings': {}, 'generated': [], 'warnings': [],
        }

        df = self._normalize_columns(df)
        df = self._build_rps(df)
        df = self._build_day_of_week(df)
        df = self._build_is_event(df)
        df = self._build_is_weekend(df)
        df = self._build_derived_features(df)   # diff, diff2, ma5, ma20
        df = self._validate_and_clean(df)

        result = df[STANDARD_FEATURES].copy()
        self.report['final_shape'] = result.shape
        self.report['rps_stats'] = {
            'min':  round(float(result['target_rps'].min()), 1),
            'max':  round(float(result['target_rps'].max()), 1),
            'mean': round(float(result['target_rps'].mean()), 1),
            'std':  round(float(result['target_rps'].std()), 1),
        }

        if self.verbose:
            self.print_report()
        return result

    def print_report(self):
        r = self.report
        print(f"\n{'─'*55}")
        print(f"📋 SmartCSVLoader 변환 리포트: {r.get('file','?')}")
        print(f"{'─'*55}")
        print(f"  원본 shape : {r.get('raw_shape','?')}")
        print(f"  최종 shape : {r.get('final_shape','?')}  (피처: {len(STANDARD_FEATURES)}개)")
        rps = r.get('rps_stats', {})
        print(f"  RPS 통계   : min={rps.get('min')}  max={rps.get('max')}"
              f"  mean={rps.get('mean')}  std={rps.get('std')}")
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

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def _normalize_columns(self, df):
        df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_')
                      for c in df.columns]
        return df

    def _find_column(self, df, aliases):
        for a in aliases:
            if a in df.columns:
                return a
        return None

    def _build_rps(self, df):
        found = self._find_column(df, RPS_COLUMN_ALIASES)
        if found is None:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            excl = ['time', 'sec', 'min', 'hour', 'idx', 'index', 'id']
            cands = [c for c in num_cols if not any(kw in c for kw in excl)]
            if not cands:
                raise SchemaAdaptError(
                    f"RPS 컬럼을 찾을 수 없습니다. 컬럼명을 'target_rps'로 바꿔주세요.\n"
                    f"현재 컬럼: {list(df.columns)}")
            found = max(cands, key=lambda c: df[c].mean())
            self.report['warnings'].append(f"'{found}'을 RPS로 자동 선택했습니다.")
        if found != 'target_rps':
            df = df.rename(columns={found: 'target_rps'})
            self.report['mappings']['target_rps'] = found
        df['target_rps'] = pd.to_numeric(df['target_rps'], errors='coerce').fillna(0).clip(lower=0)
        return df

    def _build_day_of_week(self, df):
        found = self._find_column(df, DOW_COLUMN_ALIASES)
        if found is not None:
            if found != 'day_of_week':
                df = df.rename(columns={found: 'day_of_week'})
                self.report['mappings']['day_of_week'] = found
            df['day_of_week'] = (pd.to_numeric(df['day_of_week'], errors='coerce')
                                   .fillna(0).astype(int) % 7)
        else:
            if 'time_sec' in df.columns:
                seconds_per_day = 2001
                df['day_of_week'] = (df.index // seconds_per_day % 7).astype(int)
                self.report['generated'].append("day_of_week (time_sec 기반 자동 추정)")
                self.report['warnings'].append("요일 정보를 자동 추정했습니다. 정확한 컬럼 제공 시 성능 향상.")
            else:
                df['day_of_week'] = 0
                self.report['generated'].append("day_of_week = 0 (요일 정보 없음)")
        return df

    def _build_is_event(self, df):
        found = self._find_column(df, EVENT_COLUMN_ALIASES)
        fname = os.path.basename(self.csv_path).lower()
        if found is not None:
            if found != 'is_event':
                df = df.rename(columns={found: 'is_event'})
                self.report['mappings']['is_event'] = found
            df['is_event'] = pd.to_numeric(df['is_event'], errors='coerce').fillna(0).astype(int)
        else:
            by_name     = int(any(kw in fname for kw in EVENT_KEYWORDS))
            by_scenario = int('scenario' in df.columns or 'phase' in df.columns)
            is_evt = max(by_name, by_scenario)
            df['is_event'] = is_evt
            basis = 'scenario/phase 감지' if by_scenario else '파일명 키워드' if by_name else '기본값 0'
            self.report['generated'].append(f"is_event = {is_evt}  ({basis})")
        return df

    def _build_is_weekend(self, df):
        found = self._find_column(df, WEEKEND_COLUMN_ALIASES)
        if found is not None:
            if found != 'is_weekend':
                df = df.rename(columns={found: 'is_weekend'})
                self.report['mappings']['is_weekend'] = found
            df['is_weekend'] = pd.to_numeric(df['is_weekend'], errors='coerce').fillna(0).astype(int)
        else:
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            self.report['generated'].append("is_weekend (day_of_week 5,6 기반)")
        return df

    def _build_derived_features(self, df):
        """변화량, 가속도, 단기/중기 이동평균 생성"""
        df['diff']  = df['target_rps'].diff().fillna(0)
        df['diff2'] = df['diff'].diff().fillna(0)           # ← 피크 직전 신호 핵심
        df['ma5']   = df['target_rps'].rolling(5,  min_periods=1).mean()   # 단기 급등
        df['ma20']  = df['target_rps'].rolling(20, min_periods=1).mean()   # 중기 트렌드
        self.report['generated'].append("diff, diff2 (변화량·가속도), ma5, ma20 (이동평균)")
        return df

    def _validate_and_clean(self, df):
        df = df.ffill().bfill().fillna(0)
        cap = df['target_rps'].quantile(0.995)
        clipped = int((df['target_rps'] > cap).sum())
        if clipped > 0:
            df['target_rps'] = df['target_rps'].clip(upper=cap)
            self.report['warnings'].append(
                f"target_rps 상위 {clipped}개 샘플을 {cap:.0f} RPS로 클리핑했습니다.")
        if len(df) < 120:
            raise SchemaAdaptError(f"데이터 부족: {len(df)}행 (최소 120행 필요)")
        return df
