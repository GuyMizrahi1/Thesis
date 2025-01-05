bld_NIR_mdl = function(clb) {
  # Identify columns with numeric names, assumed to be spectral data columns
  cs = which(!is.na(as.numeric(colnames(clb))))

  # Create a new DataFrame with non-spectral columns and a matrix of spectral data
  trn = data.frame(clb[, -..cs], spc = I(as.matrix(clb[, ..cs])))

  # Initialize an empty list to store the models
  ftnir_mdls = list()
  # mdl=unique(trn$model)[1]

  # Loop through each unique model identifier in the DataFrame
  for (mdl in unique(trn$model)) {

    # Filter the DataFrame to include only rows corresponding to the current model
    t1 = trn[clb[, .I[model %chin% mdl]],]

    # Skip to the next model if there are fewer than 10 non-NA values in the 'value' column
    if (dim(t1[!is.na(t1$value),])[1] < 10) next

    # Build a PLSR model using the Savitzky-Golay smoothed spectral data as predictors
    pls = plsr(value ~ savitzkyGolay(spc, 1, 2, 5), data = t1, validation = 'CV', ncomp = 10)
    # pls=plsr(value~(spc), data=t1, validation='CV', ncomp=10)

    # Select the optimal number of components for the PLSR model using the "one sigma" rule
    nc = selectNcomp(pls, 'onesigma')

    # Skip to the next model if the number of components is less than 1 or the model is null
    if (nc < 1 | is.null(pls)) next


    # Print the model identifier and the number of components
    print(paste(mdl, nc))
    # Store additional information in the PLSR model object
    pls$unit = t1$unit[1]
    pls$variable = t1$variable[1]
    pls$filter = 'nir'
    pls$nc = nc
    pls$model = paste(t1$unit[1], t1$variable[1])

    # pls$projection=NULL
    # pls$loading.weights=NULL
    # pls$validation=NULL
    # pls$fitted.values=NULL
    # pls$residuals=NULL

    # Add the PLSR model to the list of models
    ftnir_mdls[[mdl]] = pls
  }
  # Return the list of PLSR models
  ftnir_mdls
}