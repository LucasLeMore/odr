


# library #----------------------------------------------------------------------------------------------------
library(tidyverse)
library(datasets)  # importing datasets
if(require('pacman')) print ('instaled') else install.packages (pacman)

#Reticulate to import Python # -------------------------------------------------------------------------------

pacman::p_load(reticulate)

reticulate::import("proyecto completo.py")


# Load data  #----------------------------------------------------------------------------------


?iris # help about iris dataset
df <- iris
head(df)
df

#load test
test <- read_csv('prueba.csv')

test

# plot test with iris data
hist(df$Petal.Width,
     main = 'Titulo',
     xlab = 'Etiqueta x')



# Usefull comands #------------------------------------------------------------------------------

# setwd()  sets working directory
# install.packages (pacman) 


