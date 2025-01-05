bld_NIR_mdl = function(clb) {

  cs = which(!is.na(as.numeric(colnames(clb))))
  trn = data.frame(clb[, -..cs], spc = I(as.matrix(clb[, ..cs])))

  ftnir_mdls = list()
  # mdl=unique(trn$model)[1]

  for (mdl in unique(trn$model)) {

    t1 = trn[clb[, .I[model %chin% mdl]],]

    if (dim(t1[!is.na(t1$value),])[1] < 10) next

    pls = plsr(value ~ savitzkyGolay(spc, 1, 2, 5), data = t1, validation = 'CV', ncomp = 10)
    # pls=plsr(value~(spc), data=t1, validation='CV', ncomp=10)

    nc = selectNcomp(pls, 'onesigma')

    if (nc < 1 | is.null(pls)) next

    print(paste(mdl, nc))

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

    ftnir_mdls[[mdl]] = pls

  }

  ftnir_mdls
}