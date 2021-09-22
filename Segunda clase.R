


library(tidyverse)
library(datasets)  # importing datasets



# carga y preparacion de datos #


?iris # help about iris dataset
df <- iris
head(df)
df
test <- read_csv('prueba.csv')


hist(df$Petal.Width,
     main = 'Titulo',
     xlab = 'Etiqueta x')
