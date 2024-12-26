

library(keras3)
mnist <- dataset_mnist()
library(tidyverse)
mnist_digit <- mnist$train$x %>% as_tibble() %>% sample_n(30)

par(mfrow=c(3,10), mar=c(0.1,0.1,0.1,0.1))

digit <- function(input){
  
  d <- matrix(unlist(input), nrow = 28, byrow = FALSE)
  
  d <- t(apply(d, 2, rev))
  
  image(d, col=grey.colors(255), axes=FALSE)
  
}

for(i in 1:30) {
  
  digit(mnist_digit[i,])
  
}
par(mfrow=c(1,1))


x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y
dim(x_train);dim(x_test)
#reshape
x_train <- array_reshape(x_train,c(nrow(x_train),784))
x_test <- array_reshape(x_test, c(nrow(x_test),784))

# rescale
x_train <- x_train / 255
x_test <- x_test / 255
#원핫인코딩
y_train <- to_categorical(y_train, 10) #keras함수임
y_test <- to_categorical(y_test, 10)
dim(y_train)
model <- keras_model_sequential(input_shape = c(784)) |> 
  layer_dense(units = 256, activation = 'relu') |>
  layer_dropout(rate = 0.4) |>
  layer_dense(units = 128, activation = 'relu') |>
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 10, activation = 'softmax')
summary(model)

model |> compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)
history <- model |> fit(
  x_train, y_train,
  epochs=30, batch_size = 128,
  validation_split = 0.2
)
history
#test데이터에 대한 모델 성능 평가
model |> evaluate(x_test, y_test)
#예측
probs <- model |> predict(x_test)
max.col(probs)-1
