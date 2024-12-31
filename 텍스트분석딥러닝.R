library(keras3)
library(tidyverse)
library(tidymodels)
library(tidytext)
library(textrecipes)
text <- read_csv('C:\\Users\\xoosl\\Desktop\\전공\\응용통계세미나\\세미나\\Youtube-Spam-Dataset.csv')
head(text)
str(text)

word_freq <- text |> 
  unnest_tokens(word,CONTENT) |> 
  anti_join(get_stopwords()) |> 
  filter(
    str_detect(word, pattern = "^[a-zA-Z]{3,}$"),
    !str_detect(word, pattern = "^[0-9]")) |> count(CLASS,word,sort=T)
#누적빈도 계산 (max words를 위해서)
word_freq_cumsum <- word_freq |> 
  mutate(cumsum_n = cumsum(n)) |> 
  mutate(coverage=cumsum_n/sum(n)*100)
word_freq_cumsum |> filter(coverage>30) #max_words를 5000개정도?
#시퀀스 길이 계산
sequence_length <- text |> 
  unnest_tokens(word,CONTENT) |> 
  anti_join(get_stopwords()) |> 
  filter(
    str_detect(word, pattern = "^[a-zA-Z]{3,}$"),
    !str_detect(word, pattern = "^[0-9]"))
# 각 텍스트에 포함된 단어 수 계산
text_word_count <- word_freq |> 
  group_by(word) |>  # 각 텍스트(행)별로 그룹화
  summarise(word_count = n())  # 각 텍스트의 단어 수
# 최대 단어 수 (max_length) 계산
max_length <- max(text_word_count$word_count)
print(max_length)

set.seed(123456)
split <- mutate(text, CLASS=as.factor(CLASS)) |> initial_split(text, prop=0.7,strata=CLASS)
train <- training(split)
test <- testing(split)
max_words = 3248
max_length = 30


remove <- function(x) {
  !grepl("^[0-9]", x) & !grepl("^[a-zA-Z]{1,2}$",x)
}
rec <- recipe(~ CONTENT, train) |>
  step_tokenize(CONTENT) |>
  step_stopwords(CONTENT) |>
  step_tokenfilter(CONTENT,filter_fun = remove) |> 
  step_tokenfilter(CONTENT, max_tokens=max_words) |> 
  step_sequence_onehot(CONTENT,sequence_length = max_length)

rec_norm <- recipe(~CONTENT,train) |> 
  step_tokenize(CONTENT) |> 
  step_tokenfilter(CONTENT,max_tokens=max_words) |> 
  step_sequence_onehot(CONTENT,sequence_length = max_length)

spam_prep <- prep(rec)
spam_train <- bake(spam_prep,new_data=NULL,composition = "matrix")
dim(spam_train)

dense_model <- keras_model_sequential() |> 
  layer_embedding(input_dim = max_words +1,
                  output_dim = 12,
                  input_shape=max_length) |> 
  layer_flatten() |> 
  layer_dense(units=32, activation = 'relu') |> 
  layer_dense(units=1,activation='sigmoid')
dense_model
dense_model |> compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics=c('accuracy')
)
dense_history <- dense_model |> 
  fit(x=spam_train, y=train$CLASS,
      batch_size = 50,
      epochs=20,
      validation_split=0.25)
plot(dense_history)

set.seed(234)
spam_val <- validation_split(train, strata = CLASS)
spam_val

spam_analysis <- bake(spam_prep, new_data=analysis(spam_val$splits[[1]]),
                      composition = 'matrix') #훈련데이터
dim(spam_analysis)
spam_assess <- bake(spam_prep, new_data=assessment(spam_val$splits[[1]]),
                   composition = 'matrix') #valid데이터
dim(spam_assess)

class_analysis <- analysis(spam_val$splits[[1]]) %>% pull(CLASS) #결과변수
class_assess <- assessment(spam_val$splits[[1]]) %>% pull(CLASS)

dense_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1,
                  output_dim = 12,
                  input_shape = max_length) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")


dense_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
val_history <- dense_model %>%
  fit(
    x = spam_analysis,
    y = class_analysis,
    batch_size = 512,
    epochs = 10,
    validation_data = list(spam_assess, class_assess),
    verbose = FALSE
  )

val_history

keras_predict <- function(model, baked_data, response){
  predictions <- predict(model,baked_data)[,1]
  tibble(
    .pred_1 = predictions,
    .pred_class = if_else(.pred_1 <0.5, 0, 1),
    class = response
  ) |> 
    mutate(across(c(class,.pred_class),
                  ~factor(.x,levels=c(1,0))))
}
val_res <- keras_predict(dense_model, spam_assess, class_assess)
val_res

metrics(val_res, class, .pred_class)
val_res |> conf_mat(class,.pred_class)
val_res %>%
  roc_curve(truth = class, .pred_1) %>%
  autoplot() +
  labs(
    title = "Receiver operator curve for Kickstarter blurbs"
  )



#bow기능 사용

remove <- function(x) {
  !grepl("^[0-9]", x) & !grepl("^[a-zA-Z]{1,2}$",x)
}

rec <- recipe(~ CONTENT, train) |>
  step_tokenize(CONTENT) |>
  step_stopwords(CONTENT) |>
  step_tokenfilter(CONTENT,filter_fun = remove) |> 
  step_tokenfilter(CONTENT, max_tokens=1e3) |> 
  step_tf(CONTENT)


set.seed(234)
spam_val <- initial_validation_split(train, strata = CLASS)
spam_val

spam_bow_prep <- prep(rec)
spam_bow_analysis <- bake(spam_bow_prep,
                          new_data=analysis(spam_val$splits[[1]]),
                          composition = 'matrix')
spam_bow_assess <- bake(spam_bow_prep,
                        new_data = assessment(spam_val$splits[[1]]),
                        composition ='matrix')
bow_model <- keras_model_sequential() |> 
  layer_dense(units=64, activation = 'relu',input_shape=c(1e3)) |> 
  layer_dense(units=64, activation='relu') |> 
  layer_dense(units=1, activation='sigmoid')
bow_model |> compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c('accuracy')
)

bow_history <- bow_model |> 
  fit(x=spam_bow_analysis,
      y=class_analysis,
      batch_size=512,
      epochs=10,
      validation_data=list(spam_bow_assess, class_assess),
      verbose=F)
bow_history
plot(bow_history)

bow_res <- keras_predict(bow_model, spam_bow_assess, class_assess)
metrics(bow_res,.pred_class,class)
