library(tidyverse)
drought_raw <- read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-06-14/drought-fips.csv')
str(drought_raw)
#텍사스지역의 2021년 데이터만 추출
drought <- drought_raw |> 
  filter(State=='TX', lubridate::year(date) == 2021) |> 
  group_by(GEOID = FIPS) |> summarise(DSCI= mean(DSCI)) |> 
  ungroup()

library(tidycensus)#미국 인구조사국(Census Bureau)의 데이터를 R에서 쉽게 가져오고 분석할 수 있게 도와주는 R 패키지
tx_median_rent <- get_acs(
  geography = 'county',    # 데이터의 지리적 단위를 "카운티"로 설정.
  state = 'TX',            # 텍사스 주 데이터를 선택.
  variables = 'B19013_001',# "B19013_001"은 ACS 데이터에서 중위 소득(median household income)을 나타내는 변수.
  year = 2020,             # 2020년 5년간 ACS 데이터 사용.
  geometry = TRUE          # 지리 정보를 포함해 데이터를 가져옴. (지도 시각화 가능)
)

drought_sf <- tx_median_rent |> left_join(drought)
drought_sf |> ggplot(aes(fill=DSCI))+
  geom_sf(alpha=0.9, color=NA) + scale_fill_viridis_c()

drought_sf |> ggplot(aes(x=DSCI,y=estimate)) + 
  geom_point(size=2, alpha=0.8) + 
  geom_smooth(method='lm') + 
  scale_y_continuous(labels = scales::dollar_format())+ #y축 미국달러로 표시
  labs(x='Drought score',y="Median household income")

library(tidymodels)
library(spatialsample) #공간자기상관을 이용한 리샘플링 기법
set.seed(123)
folds <- spatial_block_cv(drought_sf,v=10)
autoplot(folds)
autoplot(folds$splits[[1]])

drought_res <- 
  workflow(estimate~DSCI, linear_reg()) |> 
  fit_resamples(folds, control= control_resamples(save_pred = T))
drought_res
collect_metrics(drought_res)
collect_predictions(drought_res)

drought_rmse <- 
  drought_sf |> 
  mutate(.row = row_number()) |> 
  left_join(collect_predictions(drought_res)) |> 
  group_by(GEOID) |> 
  rmse(.pred,estimate) |> 
  select(GEOID,.estimate)
drought_rmse
drought_sf |> 
  left_join(drought_rmse) |> 
  ggplot(aes(fill=.estimate)) +
  geom_sf(alpha=0.8, color=NA) + 
  labs(fill='RMSE') + 
  scale_fill_viridis_c(labels = scales::dollar_format())
