# 스마트 창고 출고 지연 예측 발표안

## 1. 프로젝트 목표
- 목표: 현재 시점 데이터로 향후 30분 평균 출고 지연(`avg_delay_minutes_next_30m`) 예측
- 데이터: `train(250,000 x 94)`, `test(50,000 x 93)`, `layout_info(300 x 15)`
- 핵심 이슈: 결측치 다수, 시나리오 단위 누수 위험, 운영 요인의 복합 비선형성

## 2. 우리가 선택한 접근
- 단순 상관계수 Top-N만 쓰지 않고, 운영 관점 카테고리 기반으로 피처 구성
- 카테고리: 배터리충전, 혼잡충돌, 주문수요, 로봇AGV, KPI성과, 인력안전, 물류재고, 환경시설, 외부날씨, 인프라
- 각 카테고리에서 2~4개 핵심 컬럼 선택 -> 총 약 33개 핵심 피처 구성

## 3. 카테고리별 피처를 이렇게 넣은 이유
### 3-1. 선정 원칙
- 원칙 1: 지연의 원인 흐름을 설명할 수 있는 피처
- 원칙 2: 같은 정보 중복(고상관 중복 피처) 최소화
- 원칙 3: 운영 의사결정에 해석 가능한 피처 우선
- 원칙 4: 결측이 있더라도 신호가 있으면 살리는 방향

### 3-2. 카테고리 예시
- 배터리충전: `low_battery_ratio`(저배터리 로봇 비율), `charge_queue_length`(충전 대기열 길이), `avg_charge_wait`(평균 충전 대기시간)
- 혼잡충돌: `congestion_score`(혼잡 점수), `max_zone_density`(구역 최대 밀집도), `near_collision_15m`(15분 내 근접충돌 건수)
- 주문수요: `order_inflow_15m`(15분 주문 유입량), `unique_sku_15m`(15분 고유 SKU 수), `urgent_order_ratio`(긴급 주문 비율), `sku_concentration`(SKU 집중도)
- 로봇AGV: `robot_utilization`(로봇 가동률), `robot_charging`(충전 중 로봇 수), `task_reassign_15m`(15분 작업 재할당 건수), `agv_task_success_rate`(AGV 작업 성공률)
- KPI성과: `kpi_otd_pct`(정시 출고율), `sort_accuracy_pct`(분류 정확도), `manual_override_ratio`(수동 개입 비율)
- 인력안전: `staff_on_floor`(현장 인력 수), `forklift_active_count`(가동 지게차 수), `safety_score_monthly`(월간 안전 점수)
- 물류재고: `loading_dock_util`(도크 활용률), `staging_area_util`(스테이징 구역 활용률), `inventory_turnover_rate`(재고 회전율), `backorder_ratio`(백오더 비율)
- 환경시설: `warehouse_temp_avg`(창고 평균 온도), `humidity_pct`(습도), `layout_compactness`(레이아웃 압축도), `maintenance_schedule_score`(정비 일정 점수)
- 외부날씨: `external_temp_c`(외부 기온), `precipitation_mm`(강수량)
- 인프라: `wms_response_time_ms`(WMS 응답시간), `network_latency_ms`(네트워크 지연), `scanner_error_rate`(스캐너 오류율)

## 4. 결측치 처리 전략
- 결론: 결측 행 삭제는 하지 않음

### 왜 삭제하지 않았나?
- 결측이 특정 컬럼에 넓게 분포 -> 삭제 시 데이터 손실 큼
- 결측 자체가 운영상 이상 상태 신호일 가능성 존재
- 트리 계열은 결측 신호를 활용 가능

### 실제 적용
- 결측률 8% 이상 피처에 `is_null_x` indicator 생성
- 수치형 결측은 중앙값 대치(학습 데이터 기준)
- 테스트에도 같은 기준 적용(데이터 누수 방지)

## 5. 이상치 처리 전략
### 왜 이상치를 제거하지 않았나?
- 창고 지연 데이터는 피크 상황이 본질(극단값 자체가 중요한 시그널)
- 단순 제거는 실전 상황 학습 능력을 떨어뜨릴 수 있음

### 실제 적용
- 수치형을 `0.5%~99.5%` 분위수로 클리핑
- 긴 꼬리 변수는 `log1p` 파생
  - `order_inflow_15m`, `charge_queue_length`, `avg_charge_wait`

## 6. 검증 전략
- `GroupKFold(groups=scenario_id)` 사용
- 이유: 같은 시나리오가 train/valid에 동시에 들어가면 점수 부풀림(누수)
- 모델: LightGBM (시드 2개 앙상블)

## 7. 현재 결과 요약
- 카테고리 모델 블렌드 OOF MAE: `9.58294`
- 참고: 더 많은 피처/파생을 사용한 강한 모델은 약 `9.14` 수준까지 가능

## 8. 점수가 낮을 때 다음에 선택할 피처
### 우선 추가 후보(1순위)
- 배터리: `battery_mean`, `battery_std`, `charge_efficiency_pct`
- 혼잡: `blocked_path_15m`, `intersection_wait_time_avg`
- 로봇: `robot_idle`, `avg_trip_distance`
- 물류재고: `replenishment_overlap`, `outbound_truck_wait_min`, `dock_to_stock_hours`
- KPI/품질: `quality_check_rate`, `barcode_read_success_rate`

### 레이아웃/시설 메타 강화(2순위)
- `pack_station_count`, `charger_count`, `robot_total`, `zone_dispersion`
- 이유: 같은 운영 상태라도 레이아웃 구조가 지연 민감도를 바꿀 수 있음

### 파생 피처 강화(3순위)
- 비율: `order_inflow_15m / (robot_active+1)`
- 상호작용: `congestion_score * order_inflow_15m`
- 압력지표: `pack_utilization * loading_dock_util`

## 9. 발표 멘트(1분 버전)
- "이번 모델은 피처를 무작정 늘리지 않고, 운영 관점 10개 카테고리로 구조화했습니다."
- "각 카테고리에서 지연 원인을 설명할 수 있는 핵심 컬럼만 선별해 해석성과 일반화 성능을 동시에 확보했습니다."
- "결측치는 삭제 대신 indicator와 중앙값 대치를 사용해 정보 손실을 줄였고, 이상치는 제거 대신 클리핑과 로그 변환으로 안정화했습니다."
- "검증은 scenario 그룹 기반 GroupKFold로 누수를 방지했고, 현재 OOF MAE는 9.58 수준입니다."
- "다음 단계는 혼잡-도크-충전 병목 관련 피처를 확장해 9.3 이하로 낮추는 것입니다."

## 10. 부족했던 부분과 보완 계획
### 부족했던 부분
- 카테고리 모델은 해석성은 좋지만 성능 한계가 존재
- 일부 컬럼 간 강한 상호작용을 충분히 반영하지 못함
- 시계열(시나리오 내 25 step) 패턴 피처가 부족

### 보완 계획
- 시나리오 내부 순서 피처(`scenario_step`) 추가
- 그룹 집계 피처(시나리오/레이아웃 단위 평균, 편차) 추가
- 모델 앙상블 강화(파라미터 다양화 + 시드 다양화)
- `layout_info` 결합 피처를 더 체계적으로 확장

## 11. 실행 체크리스트
- [ ] 카테고리 모델 기준선 점수 재현
- [ ] 추가 후보 피처 1차 반영(A/B 실험)
- [ ] 누수 없는 CV 성능 확인
- [ ] 제출 파일 생성 및 LB 확인
- [ ] LB 결과 기반 최종 피처셋 확정

## 12. 첨부 파일
- 카테고리 매핑: `code/category_feature_map.csv`
- 그룹 매핑: `code/df_col_grouped.csv`
- 카테고리 모델 노트북: `code/dacon_category_model.ipynb`
- 제출 파일: `code/submission_category.csv`
