library(data.table)
library(dplyr)
library(Matrix)
library(reshape2)
library(ggplot2)
library(pracma)
library(nlme)
library(MASS)
library(prospectr)
library(gridExtra)
library(spectacles)
library(pavo)
library(signal)
library(caret)
library(pls)
library(splitTools)
library(gt)
library(ggsci)
# remotes::install_github("philipp-baumann/simplerspec")
library(simplerspec)
library(prospectr)
library(yardstick)
## Step 0: From ASD to FTNIR<------------------------------------------------------------------------
# the first step of this flow chart is to check if the data were coming from ASD or from FTNIR devices. 
# the first step is to make a spectral resampling from ASD to FTNIR according to the followng chunk. 
# if the data were coming from  FTNIR or it is neccessary to work with ASD data, DONT run this chunk, 
# 
# setwd("C:\\Users\\dubinin\\Desktop\\FTNIR\\ASD_only")
# 
# # rewd ASD data
# ASD = read.csv("NIR_N_12.10.2023_ASD_for_Almonds.csv")
# names(ASD)
# ASD_spectral_data = ASD[,c(31:2181)]
# names(ASD_spectral_data) 
# names(ASD_spectral_data) = as.numeric(substr(names(ASD_spectral_data),2,8))
# # from wavelength to wavenumber
# names(ASD_spectral_data)  = 10000000/as.numeric(names(ASD_spectral_data))
# range(as.numeric(names(ASD_spectral_data)))
# 
# 
# # rewd FTNIR data
# FTNIR = data.table::fread("C:\\Users\\dubinin\\Desktop\\FTNIR\\81_observations\\FTNIR_data_after_correction.csv")
# names(FTNIR)
# FTNIR_spectral_data = FTNIR[,c(33:1589)]
# names(FTNIR_spectral_data)
# range(as.numeric(names(FTNIR_spectral_data)), na.rm = T)
# 
# 
# ASD_spectral_data = ASD_spectral_data[,names(ASD_spectral_data) %in% as.character(as.numeric(names(ASD_spectral_data))[as.numeric(names(ASD_spectral_data)) <= max(as.numeric(names(FTNIR_spectral_data)), na.rm = T) &
#                                                                                                                          as.numeric(names(ASD_spectral_data)) >=  min(as.numeric(names(FTNIR_spectral_data)), na.rm = T)])]
# nrow(ASD_spectral_data)
# # order the numbernumbers in ASD into 
# ASD_spectral_data = ASD_spectral_data[,rev(names(ASD_spectral_data))] # check if necessary
# # names(ASD_spectral_data) 
# # rescaling ASD to FTNIR
# wav <- as.numeric(colnames(ASD_spectral_data))
# new.wav =  as.numeric(colnames(FTNIR_spectral_data), na.rm = T)
# # increase spectral resolution by 2
# ASD_spectral_data <- prospectr::resample(ASD_spectral_data, wav, new.wav, interpol = 'spline')
# ASD_spectral_data = as.data.frame(ASD_spectral_data)
# names(ASD_spectral_data)
# # combine the resmapled data with other data coulmns 
# ASD = cbind(ASD[,c(1:30)], ASD_spectral_data) 
# 
# # write ASD resmpled data
# fwrite(ASD, "ASD_data_after_spectral_resampling.csv", row.names = F)

## Step 1: reorganization of the data <------------------------------------------------------------------------
# 1.1. check the coulmns names
setwd("D:\\OneDrive - post.bgu.ac.il\\scripts_in_work\\FTNIR_Model\\New_model")

data = read.csv("GLT_N_SPECT_FTNIR.csv", fileEncoding = "Latin1", check.names = F)
names(data)
nrow(data)
# 1.2. find the coulm of classification and change its name to Class_PLS_DA and locate it as the first coulm 
Class_PLS_DA_coulmns = 1
Class_PLS_DA = as.data.frame(data[,Class_PLS_DA_coulmns])
names(Class_PLS_DA) = names(data)[Class_PLS_DA_coulmns]

# names(Class_PLS_DA) = "Class_PLS_DA"
# 3. find the Y-s v
Y_coulmns = c(2)
Y_data = as.data.frame(data[, Y_coulmns]) 
names(Y_data) = names(data)[Y_coulmns]

# 1.4.  find X-s ariables in the data, if the X-s Variables are in wavemumber, convert it to wavelength by using the next formula: 10,000,000/wavenumber in cm 
x_coulmns = 3:8507
X_data = data[,x_coulmns]
names(X_data) = names(data)[x_coulmns]

# 1.5. set other coloums in sepeate data 
explained_coulmns = names(data)[-c(Class_PLS_DA_coulmns,Y_coulmns,x_coulmns)]
Explaines_data = data[,names(data) %in% explained_coulmns]


# 1.6. remove outliers according to Y and X data (according to MAD algoithm)

fun <- function(z, fac = 1.5 , na.rm = TRUE) {
  Q <- quantile(z, c(0.25, 0.75), na.rm = T)
  R <- Q[2] - Q[1]
  z[z < Q[1] - fac * R | z > Q[2] + fac * R] <- NA
  z
}


isnum <- sapply(X_data, is.numeric)
X_data[isnum] <- lapply(X_data[isnum], fun)

isnum <- sapply(Y_data, is.numeric)
Y_data[isnum] <- lapply(Y_data[isnum], fun)

# 1.7. set the coulms in the following order: Class_PLS_DA, Explaines_data, Y_data, X_data and save it for backup
data = cbind(Class_PLS_DA,Explaines_data,Y_data, X_data)
fwrite(data,"organized_FTNIR_data.csv", row.names = F)

## Step 2 spectral transformations <----------------------------------------------------------------------------------
#setwd("F:\\FTNIR")
data = read.csv("organized_FTNIR_data.csv")
data = as.data.frame(na.omit(object = as.data.table(data),cols = 3:8507))
names(data)
# plot spectra before transformation 
#select only the spectral data
X_data = data[,3:8507]

names(X_data)
diff(as.numeric(substr(names(X_data), 2,7)))
# names(X_data) = as.numeric(substr(names(X_data), 2,8))
signature_data = X_data
signature_data = Spectra(wl = 1:8505, nir = X_data, id = row.names(X_data), units = "nm")
plot(signature_data, gaps = T)



# 2.2. x-s transformation
# 2.2.1. Autoscale transformation
AS <- apply_spectra(signature_data, scale, center = TRUE, scale = TRUE)
plot(AS)
# 2.2.2. Mean centering transformation
MC <- apply_spectra(signature_data, scale, center = TRUE, scale = FALSE)
plot(MC)
# 2.2.3. standard normal variate (SNV) transformation
SNV <- apply_spectra(signature_data, snv)
plot(SNV)
# 2.2.4. simple savizky golay smoothing 
SGS <- apply_spectra(signature_data, sgolayfilt, n = 5, p = 3, m = 0)
plot(SGS)
# 2.2.5. 1st derivative savizky golay smoothing 
SG1D <- apply_spectra(signature_data, sgolayfilt, n = 5, p = 3, m = 1)
plot(SG1D)

# 2.2.6, 2nd derivative savizky golay smoothing 
SG2D <- apply_spectra(signature_data, sgolayfilt, n = 5, p = 3, m = 2)
plot(SG2D)
# 2.2.7, Multiplicative Scatter Correction transformation  
MSC = as.data.frame(prospectr::msc(X_data, ref_spectrum = colMeans(X_data, na.rm = T)))
names(MSC) = 1:8505
spectacles::spectra(MSC) <- ~ 1:8505
plot(MSC)
# 2.2.8. GLSW transformation
cov = cov(X_data) 
comp = svd(cov)
eignevales = diag(comp$d)
eignevectors = comp$v
eignevectors_transpose = comp$u
D = sqrt((eignevales/0.02)+diag(dim(eignevales)[1]))
G = eignevectors %*%  MASS::ginv(D) %*% t(eignevectors_transpose)
GLSW = t(G %*% t(X_data))
GLSW = as.data.frame(GLSW)
names(GLSW) = 1:8505

spectacles::spectra(GLSW) <-~ 1:8505
plot(GLSW)
# 2.2.9. SG1D+SNV
SG1D_SNV = apply_spectra(SG1D, snv)
plot(SG1D_SNV)
# 2.2.10. MSC+SG2D+MC
MCS_SG2D_MC = apply_spectra((apply_spectra(MSC,sgolayfilt, n = 5, p = 3, m = 2)), scale, center = TRUE, scale = FALSE)
plot(MCS_SG2D_MC)
# 2.2.11. SGS+AS+GLSW
SGS_AS_GLSW = apply_spectra(SGS,scale, center = TRUE, scale = TRUE)
SGS_AS_GLSW = as.data.frame(SGS_AS_GLSW)
cov = cov(SGS_AS_GLSW[,2:8506]) 
comp = svd(cov)
eignevales = diag(comp$d)
eignevectors = comp$v
eignevectors_transpose = comp$u
D = sqrt((eignevales/0.02)+diag(dim(eignevales)[1]))
G = eignevectors %*%  MASS::ginv(D) %*% t(eignevectors_transpose)
SGS_AS_GLSW = t(G %*% t(SGS_AS_GLSW[,2:8506]))
SGS_AS_GLSW = as.data.frame(SGS_AS_GLSW)
names(SGS_AS_GLSW) = 1:8505
spectacles::spectra(SGS_AS_GLSW) <- ~ 1:8505
plot(SGS_AS_GLSW)




## Step 3: prepossessing  <---------------------------------------------------
# 3.1 Seperate the data into cal, val and pred 
# Remember! we will use Y_data, X_data and its transformations (AS,MC,SNV,SGS,SG1D,
#SG2D, MSC,GLSW, SG1D_SNV,MCS_SG2D_MC, SGS_AS_GLSW ) for pls-R analysis. 

## Step 4:the models
# 4,1 Creating the models 

Y_data = as.data.frame(data[,2])
names(Y_data) = names(data)[2]
fun1 <- function(z, na.rm = TRUE) {
  z[z == ""] <- NA
  z[z == "N/D"] <- NA
  z = as.numeric(z)
  if(length(!is.na(z)) <= 30){
    z = NULL
  }
}

isnum <- sapply(Y_data, is.character)
Y_data[isnum] <- lapply(Y_data[isnum], fun1)

# remove the colmns with same value
Y_data = Y_data[vapply(Y_data, function(x) length(unique(x)) > 2, logical(1L))]



set.seed(54455)

objs =  mget(ls(envir=.GlobalEnv), envir=.GlobalEnv)
objs = Filter(function(i) inherits(i, "Spectra"), objs)
names_tran = names(Filter(function(i) inherits(i, "Spectra"), objs))

obspred = NULL

# for(i in 1:ncol(Y_data)){  # check what was happended with i=12
#   data_summary = NULL
#   predictions_calibration =  vector(mode='list', length=length(objs))
#   predictions_validation =  vector(mode='list', length=length(objs))
#   
#   observation_calibration =  vector(mode='list', length=length(objs))
#   observation_validation =  vector(mode='list', length=length(objs))
#   prediction_data =  vector(mode='list', length=length(objs))
#   
#   models =  vector(mode='list', length=length(objs))
#   
#   for(j in 1:length(objs)){
#     chem = as.data.frame(cbind(1:nrow(Y_data),data[,1:2],Y_data[,i]))
#     names(chem) = c("sample_id",names(data)[1], names(data)[2], names(data)[3])
#     chem$sample_id = as.character(chem$sample_id)
#     chem = as_tibble(chem)
#     spec = as_tibble(objs[[j]])  
#     names(spec)[1]= "sample_id"
#     spec_chem = simplerspec::join_spc_chem(spc_tbl = spec ,chem_tbl = chem,  by = "sample_ID")
#     spec_chem = spec_chem[complete.cases(spec_chem), ]
#     inds <- partition(spec_chem[[names(spec_chem)[8509]]], p = c(train = 0.7*0.8, valid = 0.3*0.8, test = 0.2))
#     
#     train.data  <- spec_chem[inds$train, -1]
#     test.data <- spec_chem[inds$valid, -1]
#     pred.data <- spec_chem[inds$test, -1]
#     
#     model <- train(x = train.data[,-c(8506:8508)], y = as.numeric(train.data[[names(train.data)[8508]]]),
#                    method = "pls",
#                    metric =  "RMSE",
#                    trControl = trainControl(method = "cv"),
#                    tuneLength = 5)
#     
#     
#     models[[j]] = model
#     # minimize the cross-validation error, RMSE
#     
#     predictions_calibration[[j]] <- model %>% predict(train.data)
#     predictions_validation[[j]] <- model %>% predict(test.data)
#     
#     observation_calibration[[j]] =  as.numeric(train.data[[names(test.data)[8508]]])
#     observation_validation[[j]] =  as.numeric(test.data[[names(test.data)[8508]]])
#     prediction_data[[j]] =  pred.data
#     
#     
#     c = data.frame(
#       transform = names_tran[j],
#       RMSE = caret::RMSE(predictions_validation[[j]] ,  observation_validation[[j]]),
#       Rsquare = caret::R2(predictions_validation[[j]],  observation_validation[[j]])
#     )
#     data_summary = rbind(data_summary, c)
#   }
#   write.csv(data_summary, paste0(names(Y_data)[i],"transformations_models.csv"), row.names = F)
#   best_model = data_summary[data_summary$RMSE == min(data_summary$RMSE, na.rm = T) |
#                               data_summary$Rsquare == max(data_summary$Rsquare, na.rm = T),]
#   if(nrow(best_model) >1){
#     best_model = best_model[best_model$RMSE == min(best_model$RMSE, na.rm = T),]
#   }
#   
#   d = as.data.frame(c(observation_calibration[[as.numeric(row.names(best_model))]]))
#   names(d) = "observed"
#   prediction = as.data.frame(c(predictions_calibration[[as.numeric(row.names(best_model))]]))
#   names(prediction) = "predicted"
#   d = cbind(d, prediction)
#   d$train_test = "calibration"
#   d$variable = names(Y_data)[i]
#   
#   v = as.data.frame(c(observation_validation[[as.numeric(row.names(best_model))]]))
#   names(v) = "observed"
#   prediction_v = (as.data.frame(c(predictions_validation[[as.numeric(row.names(best_model))]])))
#   names(prediction_v) = "predicted"
#   v = cbind(v, prediction_v)
#   v$train_test = "validation"
#   v$variable = names(Y_data)[i]
#   
#   data_t = as_tibble(rbind(d,v), rownames = NULL)
#   names(data_t)
#   
#   predictions <- models[[as.numeric(row.names(best_model))]] %>% predict(prediction_data[[as.numeric(row.names(best_model))]])
#   
#   saveRDS(models[[as.numeric(row.names(best_model))]], paste0(names(Y_data)[i]))
#   
#   observations <- prediction_data[[as.numeric(row.names(best_model))]][8508]
#   
#   dat = data.frame(observed =observations, 
#                    predicted = predictions,train_test = "prediction", variable = names(Y_data)[i])
#   names(dat) = names(data_t)
#   
#   data_t = as_tibble(rbind(data_t,dat), rownames = NULL)
#   
#   
#   obspred = rbind(obspred, data_t)
#   print(paste0(i, "/", ncol(Y_data)))
# }
for(i in 1:ncol(Y_data)){  # check what was happended with i=12
  data_summary = NULL
  predictions_calibration =  vector(mode='list', length=length(objs))
  predictions_validation =  vector(mode='list', length=length(objs))
  
  observation_calibration =  vector(mode='list', length=length(objs))
  observation_validation =  vector(mode='list', length=length(objs))
  prediction_data =  vector(mode='list', length=length(objs))
  
  models =  vector(mode='list', length=length(objs))
  
  for(j in 1:length(objs)){
    chem = as.data.frame(cbind(1:nrow(Y_data),data[,1:2],Y_data[,i]))
    names(chem) = c("sample_id",names(data)[1], names(data)[2], names(data)[3])
    chem$sample_id = as.character(chem$sample_id)
    chem = as_tibble(chem)
    spec = as_tibble(objs[[j]])  
    names(spec)[1]= "sample_id"
    spec_chem = simplerspec::join_spc_chem(spc_tbl = spec ,chem_tbl = chem,  by = "sample_ID")
    spec_chem = spec_chem[complete.cases(spec_chem), ]
    inds <- partition(spec_chem[[names(spec_chem)[8509]]], p = c(train = 0.7*0.8, valid = 0.3*0.8, test = 0.2))
    
    train.data  <- spec_chem[inds$train, -1]
    test.data <- spec_chem[inds$valid, -1]
    pred.data <- spec_chem[inds$test, -1]
    
    model <- train(x = train.data[,-c(8506:8508)], y = as.numeric(train.data[[names(train.data)[8508]]]),
                   method = "pls",
                   metric =  "RMSE",
                   trControl = trainControl(method = "cv"),
                   tuneLength = 5)
    
    
    models[[j]] = model
    # minimize the cross-validation error, RMSE
    
    predictions_calibration[[j]] <- model %>% predict(train.data)
    predictions_validation[[j]] <- model %>% predict(test.data)
    
    observation_calibration[[j]] =  as.numeric(train.data[[names(test.data)[8508]]])
    observation_validation[[j]] =  as.numeric(test.data[[names(test.data)[8508]]])
    prediction_data[[j]] =  pred.data
    
    
    c = data.frame(
      transform = names_tran[j],
      RMSE = caret::RMSE(predictions_validation[[j]] ,  observation_validation[[j]]),
      Rsquare = caret::R2(predictions_validation[[j]],  observation_validation[[j]])
    )
    data_summary = rbind(data_summary, c)
  }
  write.csv(data_summary, paste0(names(Y_data)[i],"transformations_models.csv"), row.names = F)
  best_model = data_summary[data_summary$RMSE == min(data_summary$RMSE, na.rm = T) |
                              data_summary$Rsquare == max(data_summary$Rsquare, na.rm = T),]
  if(nrow(best_model) >1){
    best_model = best_model[best_model$RMSE == min(best_model$RMSE, na.rm = T),]
  }
  
  d = as.data.frame(c(observation_calibration[[as.numeric(row.names(best_model))]]))
  names(d) = "observed"
  prediction = as.data.frame(c(predictions_calibration[[as.numeric(row.names(best_model))]]))
  names(prediction) = "predicted"
  d = cbind(d, prediction)
  d$train_test = "calibration"
  d$variable = names(Y_data)[i]
  
  v = as.data.frame(c(observation_validation[[as.numeric(row.names(best_model))]]))
  names(v) = "observed"
  prediction_v = (as.data.frame(c(predictions_validation[[as.numeric(row.names(best_model))]])))
  names(prediction_v) = "predicted"
  v = cbind(v, prediction_v)
  v$train_test = "validation"
  v$variable = names(Y_data)[i]
  
  data_t = as_tibble(rbind(d,v), rownames = NULL)
  names(data_t)
  
  predictions <- models[[as.numeric(row.names(best_model))]] %>% predict(prediction_data[[as.numeric(row.names(best_model))]])
  
  saveRDS(models[[as.numeric(row.names(best_model))]], paste0(names(Y_data)[i]))
  
  observations <- prediction_data[[as.numeric(row.names(best_model))]][8508]
  
  dat = data.frame(observed =observations, 
                   predicted = predictions,train_test = "prediction", variable = names(Y_data)[i])
  names(dat) = names(data_t)
  
  data_t = as_tibble(rbind(data_t,dat), rownames = NULL)
  
  
  obspred = rbind(obspred, data_t)
  print(paste0(i, "/", ncol(Y_data)))
}

## Step 4: plot results  <---------------------------------------------------

names(obspred)
str(obspred)
# obspred$observed = as.numeric(obspred$observed)

obspred$id = 1:nrow(obspred)

obspred_rm_outliers = obspred #[!obspred$id %in% c("10","21","24","90","14","91","15", "37","58","5","48","49","94","50","74"),]
ggplot(obspred_rm_outliers, aes(x = observed , y = predicted  )) +
  geom_point(size =3, aes(color = train_test)) +
  geom_smooth(aes(color =train_test , linetype = train_test ), method = "lm", size =2, se = T) +
  facet_wrap(variable~., nrow = 5, ncol = 5, scales = "free")+
  geom_text(aes(label= id))+
  theme_test() +
  theme(
    axis.title.x = element_text(size = 10,  face = "bold"),
    axis.text.x = element_text(size = 10,  face = "bold"),
    axis.text.y = element_text(size = 10,  face = "bold"),
    axis.title.y = element_text(size = 10,  face = "bold"),
    strip.text = element_text(size = 10,  face = "bold"),
    legend.title = element_text(size = 10,  face = "bold"),
    
    strip.text.y = element_text(
      size = 10, color = "black", face = "bold"
    ),
    legend.text = element_text(
      size = 10,  face = "bold"
    ))+scale_color_d3() +
  xlab("Observation") + ylab("Prediction") 


summerized_data = obspred_rm_outliers %>% group_by(variable, train_test) %>% summarise( RMSE = caret::RMSE(predicted , 
                                                                                                           observed),
                                                                                        Rsquare = caret::R2(predicted,  
                                                                                                            observed),
                                                                                        RPD = yardstick::rpd_vec(truth = observed,
                                                                                                                 estimate = predicted),
                                                                                        RPIQ = yardstick::rpiq_vec(truth = observed, 
                                                                                                                   estimate = predicted)
                                                                                        
)
summerized_data
model = readRDS("data[, 2]")
caret::varImp(model)


data = read.csv("organized_FTNIR_data.csv")
data = as.data.frame(na.omit(object = as.data.table(data),cols = 33:1589))
dat = data[,33:1589]
names(dat) = 1:1557
names(dat) = paste0("X", names(dat))
signature_data = dat
signature_data = Spectra(wl = 1:1557, nir = signature_data, id = row.names(signature_data), units = "nm")
SGS <- apply_spectra(signature_data, sgolayfilt, n = 5, p = 3, m = 0)

SGS_AS_GLSW = apply_spectra(SGS,scale, center = TRUE, scale = TRUE)
SGS_AS_GLSW = as.data.frame(SGS_AS_GLSW)
cov = cov(SGS_AS_GLSW[,2:1558]) 
comp = svd(cov)
eignevales = diag(comp$d)
eignevectors = comp$v
eignevectors_transpose = comp$u
D = sqrt((eignevales/0.02)+diag(dim(eignevales)[1]))
G = eignevectors %*%  MASS::ginv(D) %*% t(eignevectors_transpose)
SGS_AS_GLSW = t(G %*% t(SGS_AS_GLSW[,2:1558]))
SGS_AS_GLSW = as.data.frame(SGS_AS_GLSW)
names(SGS_AS_GLSW) = 1:1557
spectacles::spectra(SGS_AS_GLSW) <- ~ 1:1557
plot(SGS_AS_GLSW)

dat = as_tibble(SGS_AS_GLSW)
data = data[,c(1:32)]
data$N_predicted = predict(model, dat)

write.csv(data,"data_with_predicted_N_summarized.csv")


clib_data = read.csv("Or_model_ordered.csv")
clib_data$Instrument.sample.. = paste0(" ",clib_data$Tree.number,"_", clib_data$Flight.date,"_", clib_data$location)
clib_data = clib_data[,c(6,4,5)]

data_final = left_join(data,clib_data, by = "Instrument.sample..")


data_final$id = 1:nrow(data_final)
data_final = data_final[-c(1490,1351,1389),]
ggplot(data_final, aes(x = Analytical.value..N. , y = second.model  )) +
  geom_point(size =3, aes(color = Variety.cultivar )) +
  geom_smooth(method = "lm", size =2, se = T) +
  geom_abline(slope = 1,intercept = 0 ,size =2) +
  # geom_text(aes(label= id))+
  theme_test() +
  theme(
    axis.title.x = element_text(size = 10,  face = "bold"),
    axis.text.x = element_text(size = 10,  face = "bold"),
    axis.text.y = element_text(size = 10,  face = "bold"),
    axis.title.y = element_text(size = 10,  face = "bold"),
    strip.text = element_text(size = 10,  face = "bold"),
    legend.title = element_text(size = 10,  face = "bold"),
    
    strip.text.y = element_text(
      size = 10, color = "black", face = "bold"
    ),
    legend.text = element_text(
      size = 10,  face = "bold"
    ))+scale_color_d3() +
  xlab("Observation") + ylab("Prediction") 


fit = lm(data = data_final, N_predicted ~ second.model)
summary(fit)
caret::RMSE(pred = data_final$second.model, obs = data_final$Analytical.value..N., na.rm = T)

fit = lm(data = data_final, N_predicted ~ second.model)
summary(fit)
fit = lm(data = data_final, Analytical.value..N. ~ N_predicted)
summary(fit)
caret::RMSE(pred = data_final$N_predicted, obs = data_final$Analytical.value..N., na.rm = T)


write.csv(data_final, "data_final_compared_models.csv", row.names = F)
